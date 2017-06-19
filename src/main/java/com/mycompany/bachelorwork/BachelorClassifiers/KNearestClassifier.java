/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.bachelorwork.BachelorClassifiers;
import java.io.StringReader;
import moa.classifiers.AbstractClassifier;
import moa.core.InstancesHeader;
import moa.core.Measurement;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.KDTree;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
/**
 *
 * @author MDNote
 */
public class KNearestClassifier extends AbstractClassifier  {
	public IntOption kOption = new IntOption( "k", 'k', "The number of neighbors", 5, 1, Integer.MAX_VALUE);
  public IntOption limitOption = new IntOption( "limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);
  public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption(
    "nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{
        "LinearNN", "KDTree"},
    new String[]{"Brute force search algorithm for nearest neighbour search. ",
        "KDTree search algorithm for nearest neighbour search"
    }, 0);
  protected weka.core.Instances window;
  int C = 0;
  @Override
  public String getPurposeString() {
   return "kNN: special.";
  }
  @Override
  public void resetLearningImpl() {
    this.window  = null;
		}
  @Override
  public void trainOnInstanceImpl(Instance inst) {
    if (inst.classValue()>C){
      C=(int)inst.classValue();
    }
    if (this.window==null){
      this.window=new Instances(inst.dataset());
    }
    if (this.limitOption.getValue()<=this.window.numInstances()){
    this.window.delete(0);
    }
    this.window.add(inst);
  }
  @Override
  protected Measurement[] getModelMeasurementsImpl() {
    return null;
  }

  @Override
  public void getModelDescription(StringBuilder sb, int i) {
  }
  @Override
  public boolean isRandomizable() {
   return false;
  }
  @Override
  public double[] getVotesForInstance(Instance inst) {
    double v[] = new double[C+1];
    try {
      NearestNeighbourSearch search;
      if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
        search = new LinearNNSearch(this.window);  
      } else {
        search = new KDTree();
        search.setInstances(this.window);
      }   
      if (this.window.numInstances()>0) { 
        Instances neighbours = search.kNearestNeighbours(inst,Math.min(kOption.getValue(),this.window.numInstances()));
        for(int i = 0; i < neighbours.numInstances(); i++) {
        v[(int)neighbours.instance(i).classValue()]++;            
        }
      }
    } catch(Exception e) {
      return new double[inst.numClasses()];
    }
    return v;  
  } 
}
