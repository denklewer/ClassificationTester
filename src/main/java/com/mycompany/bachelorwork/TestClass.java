package com.mycompany.bachelorwork;


import com.mycompany.bachelorwork.BachelorClassifiers.HoeffdingClassifier;
import java.util.HashSet;
import java.util.Set;
import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.classifiers.bayes.NaiveBayes;
import moa.streams.generators.RandomRBFGenerator;
import moa.streams.ArffFileStream;
import weka.core.Instance;
import com.mycompany.bachelorwork.BachelorClassifiers.KNearestClassifier;
import com.mycompany.bachelorwork.BachelorEvaluation.BachelorClassificationEvaluator;
import com.mycompany.bachelorwork.BachelorClassifiers.HoeffdingOptionClassifier;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import moa.classifiers.trees.HoeffdingOptionTree;
import moa.core.Measurement;
import moa.evaluation.SilhouetteCoefficient;
import moa.evaluation.WindowClassificationPerformanceEvaluator;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author MDNote
 */
public class TestClass { 
  
  private static void exists(String fileName) throws FileNotFoundException {
    File file = new File(fileName);
    if (!file.exists()){
      throw new FileNotFoundException(file.getName());
    }
  }
  public static void write(String fileName, String text) {
    //Определяем файл
    File file = new File(fileName);
    try {
      //проверяем, что если файл не существует то создаем его
      if(!file.exists()){
        file.createNewFile();
      }
      //PrintWriter обеспечит возможности записи в файл
      PrintWriter out = new PrintWriter(file.getAbsoluteFile());
      try {
        //Записываем текст у файл
        out.print(text);
      } finally {
        //После чего мы должны закрыть файл
        //Иначе файл не запишется
        out.close();
      }
    } catch(IOException e) {
      throw new RuntimeException(e);
    }
  }
  public static String read(String fileName) throws FileNotFoundException {
    //Этот спец. объект для построения строки
    StringBuilder sb = new StringBuilder();
    exists(fileName);
    //Определяем файл
    File file = new File(fileName);
    try {
      //Объект для чтения файла в буфер
      BufferedReader in = new BufferedReader(new FileReader( file.getAbsoluteFile()));
      try {
        //В цикле построчно считываем файл
        String s;
        while ((s = in.readLine()) != null) {
          sb.append(s);
          sb.append("\n");
        }
      } finally {
        //Также не забываем закрыть файл
        in.close();
      }
    } catch(IOException e) {
      throw new RuntimeException(e);
    }
    //Возвращаем полученный текст с файла
    return sb.toString();
  }
  public static void update(String nameFile, String newText) throws FileNotFoundException {
    exists(nameFile);
    StringBuilder sb = new StringBuilder();
    String oldFile = read(nameFile);
    sb.append(oldFile);
    sb.append(newText);
    write(nameFile, sb.toString());
  }
  public static void main(String[] args) throws FileNotFoundException {
    String str="Finished!";
    long SumTrainingTime=0;
    long SumTestTime =0;
    //***************************************************************************************************************************
    // Основной тестовый метод   
    //***************************************************************************************************************************        
    String filename1 = "D:\\Bachelor-work\\resultOptionHoeffding2Acc.txt";
    String filename2 = "D:\\Bachelor-work\\resultOptionHoeffding2Kappa.txt";
    /*
    String filename1 = "D:\\Bachelor-work\\resultOptionHoeffding1Acc.txt";
    String filename2 = "D:\\Bachelor-work\\resultOptionHoeffding1Kappa.txt";   
    String filename1 = "D:\\Bachelor-work\\resultOptionHoeffding0Acc.txt";
    String filename2 = "D:\\Bachelor-work\\resultOptionHoeffding0Kappa.txt";
    
    String filename1 = "D:\\Bachelor-work\\resultHoeffding2Acc.txt";
    String filename2 = "D:\\Bachelor-work\\resultHoeffding2Kappa.txt";
    String filename1 = "D:\\Bachelor-work\\resultHoeffding1Acc.txt";
    String filename2 = "D:\\Bachelor-work\\resultHoeffding1Kappa.txt";
    String filename1 = "D:\\Bachelor-work\\resultHoeffding0Acc.txt";
    String filename2 = "D:\\Bachelor-work\\resultHoeffding0Kappa.txt";
    
    String filename1 = "D:\\Bachelor-work\\resultKNearestAcc.txt";
    String filename2 = "D:\\Bachelor-work\\resultKNearestKappa.txt"
    */
    write(filename1,"");
    write(filename2,"");
    int numInstances =10000;
    int i = 0;
    Classifier  learner2= new HoeffdingOptionClassifier();
    // Classifier  learner2= new KNearestClassifier();
    //Classifier  learner2= new HoeffdingClassifier();
    ((HoeffdingOptionClassifier)learner2).leafpredictionOption.setChosenIndex(2);
    System.out.println("Hoeffding 2");
    BachelorClassificationEvaluator  evaluator = new BachelorClassificationEvaluator();  
    ArffFileStream strm = new ArffFileStream("D:\\Bachelor-work\\DATA\\KDDCup9910p.arff",42);
    ArffFileStream TestStream = new ArffFileStream("D:\\Bachelor-work\\corrected\\corrected.arff",42);      
    strm.prepareForUse();
    learner2.setModelContext(strm.getHeader());
    learner2.prepareForUse();       
    int numberSamplesCorrect=0;
    int numberSamples=0;
    int numberTrainSamples=0;  
    while (strm.hasMoreInstances()) {
    i++;
    long start = System.nanoTime();
    System.out.println("number of instances for training " + numInstances);
    System.out.print(i+") ");
          
    while(strm.hasMoreInstances() && numberTrainSamples<numInstances){
        Instance trainInst = strm.nextInstance();
        learner2.trainOnInstance(trainInst);
        numberTrainSamples++;
      }              
      long end = System.nanoTime();
      SumTrainingTime+=end-start;
      System.out.println("test");
      numberTrainSamples =0;  
      evaluator.reset(39);
      TestStream.restart();
      start = System.nanoTime();
       while (TestStream.hasMoreInstances()){
        Instance testInst = TestStream.nextInstance();           
        if (learner2.correctlyClassifies(testInst)){
          numberSamplesCorrect++;
        }
        numberSamples++;
        evaluator.addResult(testInst, learner2.getVotesForInstance(testInst));
      }
      end = System.nanoTime();
      SumTestTime+=end-start;
      Measurement[] x =evaluator.getPerformanceMeasurements();
      double accuracy = 100.0*(double) numberSamplesCorrect/(double) numberSamples;
      update(filename1,evaluator.getFractionCorrectlyClassified() + "\n");
      update(filename2,evaluator.getKappaStatistic()+ "\n");
      System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy " );
      System.out.println("вес " + evaluator.TotalweightObserved);
      numberSamples=0;
      numberSamplesCorrect=0;
    }
    System.out.println(str);    
    System.out.println("Avg training time :" + SumTrainingTime/i);
    System.out.println("Avg test time :" + SumTestTime/i);
  }
}
