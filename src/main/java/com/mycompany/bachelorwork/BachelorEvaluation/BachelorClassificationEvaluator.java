
package com.mycompany.bachelorwork.BachelorEvaluation;
 
 import moa.core.Measurement;
 import moa.core.ObjectRepository;
import moa.evaluation.ClassificationPerformanceEvaluator;
 import moa.options.AbstractOptionHandler;
 import moa.options.IntOption;
 import moa.tasks.TaskMonitor;
 import weka.core.Utils;
 import weka.core.Instance;
/**
 *
 * @author MDNote
 */
public class BachelorClassificationEvaluator extends AbstractOptionHandler      implements ClassificationPerformanceEvaluator {
     public IntOption widthOption = new IntOption("width",
             'w', "Size of Window", 310000);
 
     public double TotalweightObserved = 0;
 
     protected Estimator weightObserved;
 
     protected Estimator weightCorrect;
 
     protected Estimator[] columnKappa;
 
     protected Estimator[] rowKappa;
 
     protected int numClasses;
 
     public class Estimator {
 
         protected double[] window;
 
         protected int posWindow;
 
         protected int lenWindow;
 
         protected int SizeWindow;
 
         protected double sum;
 
         public Estimator(int sizeWindow) {
             window = new double[sizeWindow];
             SizeWindow = sizeWindow;
             posWindow = 0;
         }
 
         public void add(double value) {
             sum -= window[posWindow];
             sum += value;
             window[posWindow] = value;
             posWindow++;
             if (posWindow == SizeWindow) {
                 posWindow = 0;
             }
         }
 
         public double total() {
             return sum;
         }
     }
 
     /*   public void setWindowWidth(int w) {
     this.width = w;
     reset();
     }*/
     @Override
     public void reset() {
         reset(this.numClasses);
     }
 
     public void reset(int numClasses) {
         this.numClasses = numClasses;
         this.rowKappa = new Estimator[numClasses];
         this.columnKappa = new Estimator[numClasses];
         for (int i = 0; i < this.numClasses; i++) {
             this.rowKappa[i] = new Estimator(this.widthOption.getValue());
             this.columnKappa[i] = new Estimator(this.widthOption.getValue());
         }
         this.weightCorrect = new Estimator(this.widthOption.getValue());
         this.weightObserved = new Estimator(this.widthOption.getValue());
         this.TotalweightObserved = 0;
     }
 
     @Override
     public void addResult(Instance inst, double[] classVotes) {
         double weight = inst.weight();
         int trueClass = (int) inst.classValue();
         if (weight > 0.0) {
             if (TotalweightObserved == 0) {
                 reset(inst.dataset().numClasses());
             }
             this.TotalweightObserved += weight;
             this.weightObserved.add(weight);
             int predictedClass = Utils.maxIndex(classVotes);
             if (predictedClass == trueClass) {
                 this.weightCorrect.add(weight);
             } else {
                 this.weightCorrect.add(0);
             }
             //Add Kappa statistic information
             for (int i = 0; i < this.numClasses; i++) {
                 this.rowKappa[i].add(i == predictedClass ? weight : 0);
                 this.columnKappa[i].add(i == trueClass ? weight : 0);
             }
 
         }
     }

     @Override
     public Measurement[] getPerformanceMeasurements() {
         return new Measurement[]{
                     new Measurement("classified instances",
                     this.TotalweightObserved),
                     new Measurement("classifications correct (percent)",
                     getFractionCorrectlyClassified() * 100.0),
                     new Measurement("Kappa Statistic (percent)",
                     getKappaStatistic() * 100.0)};
 
     }
 
     public double getTotalWeightObserved() {
         return this.weightObserved.total();
     }
 
     public double getFractionCorrectlyClassified() {
         return this.weightObserved.total() > 0.0 ? (double) this.weightCorrect.total()
                 / this.weightObserved.total() : 0.0;
     }
 
     public double getKappaStatistic() {
         if (this.weightObserved.total() > 0.0) {
             double p0 = this.weightCorrect.total() / this.weightObserved.total();
             double pc = 0;
             for (int i = 0; i < this.numClasses; i++) {
                 pc += (this.rowKappa[i].total() / this.weightObserved.total())
                         * (this.columnKappa[i].total() / this.weightObserved.total());
             }
             return (p0 - pc) / (1 - pc);
         } else {
             return 0;
         }
     }
 
     public double getFractionIncorrectlyClassified() {
         return 1.0 - getFractionCorrectlyClassified();
     }
 
     @Override
     public void getDescription(StringBuilder sb, int indent) {
         Measurement.getMeasurementsDescription(getPerformanceMeasurements(),
                 sb, indent);
     }
 
     @Override
     public void prepareForUseImpl(TaskMonitor monitor,
             ObjectRepository repository) {
     }
 }

