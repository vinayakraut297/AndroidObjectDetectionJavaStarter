package com.example.objectdetection;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.Math.min;

class Classifier {
    private Interpreter interpreter;
    private List<String> labelList;
    private int INPUT_SIZE ;
    private int PIXEL_SIZE = 3;
    private int IMAGE_MEAN = 0;
    private float IMAGE_STD = 127.0f;
    private float MAX_RESULTS = 1; 
    private float THRESHOLD = 0.3f;
    private boolean quantized = false;
    Classifier(AssetManager assetManager, String modelPath, String labelPath, int inputSize,boolean quantized) throws IOException {
        INPUT_SIZE = inputSize;
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(5);
        options.setUseNNAPI(true);
        this.quantized = quantized;
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
        labelList = loadLabelList(assetManager, labelPath);
    }


    public class Recognition {

        private final String id;
        private final String title;
        private final Float confidence;
        private RectF location;

        public Recognition(final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }


    private MappedByteBuffer loadModelFile(AssetManager assetManager, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    private float[][][] outputLocations;
    private float[][] outputClasses;
    private float[][] outputScores;
    private float[] numDetections;
    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }


    List<Recognition> recognizeImage(Bitmap bitmap) {
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
        outputLocations = new float[1][10][4];
        outputClasses = new float[1][10];
        outputScores = new float[1][10];
        numDetections = new float[1];

        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);
        Object[] inputArray = {byteBuffer};
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

        // Sort the results by confidence in descending order
        List<Recognition> detections = new ArrayList<>();
        for (int i = 0; i < numDetections[0]; ++i) {
            if (outputScores[0][i] > THRESHOLD) {
                final RectF detection = new RectF(
                        outputLocations[0][i][1] * bitmap.getWidth(),
                        outputLocations[0][i][0] * bitmap.getHeight(),
                        outputLocations[0][i][3] * bitmap.getWidth(),
                        outputLocations[0][i][2] * bitmap.getHeight());
                detections.add(new Recognition(
                        "" + i,
                        labelList.get((int) outputClasses[0][i]),
                        outputScores[0][i],
                        detection));
            }
        }

        // Sort by confidence and take the top results
        Collections.sort(detections, (o1, o2) -> Float.compare(o2.getConfidence(), o1.getConfidence()));

        return detections.subList(0, Math.min(detections.size(), (int) MAX_RESULTS));
    }


    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        if(quantized){
            byteBuffer = ByteBuffer.allocateDirect(1  * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        }else{
            byteBuffer = ByteBuffer.allocateDirect(4  * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                if(quantized){
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                }else{
                    byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                }
            }
        }
        return byteBuffer;
    }

//    @SuppressLint("DefaultLocale")
//    private List<Recognition> getSortedResultFloat(float[][] labelProbArray) {
//
//        PriorityQueue<Recognition> pq =
//                new PriorityQueue<>(
//                        (int) MAX_RESULTS,
//                        new Comparator<Recognition>() {
//                            @Override
//                            public int compare(Recognition lhs, Recognition rhs) {
//                                return Float.compare(rhs.confidence, lhs.confidence);
//                            }
//                        });
//
//        for(int i = 0; i < labelList.size(); ++i) {
//            float confidence = labelProbArray[0][i];
//            Log.d("tryCon",confidence+"");
//            if (confidence > THRESHOLD) {
////                pq.add(new Recognition(""+ i,
////                        labelList.size() > i ? labelList.get(i) : "unknown",
////                        confidence));
//            }
//        }
//
//        final ArrayList<Recognition> recognitions = new ArrayList<>();
//        int recognitionsSize = (int) Math.min(pq.size(), MAX_RESULTS);
//        for (int i = 0; i < recognitionsSize; ++i) {
//            recognitions.add(pq.poll());
//        }
//
//        return recognitions;
//    }

}
