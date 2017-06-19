
 package com.mycompany.bachelorwork.BachelorClassifiers;
 
 import java.util.Arrays;
 import java.util.Comparator;
 import java.util.HashSet;
 import java.util.LinkedList;
 import java.util.List;
 import java.util.Set;
 
 import moa.AbstractMOAObject;
 import moa.classifiers.AbstractClassifier;
 import moa.classifiers.bayes.NaiveBayes;
 import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
 import moa.classifiers.core.AttributeSplitSuggestion;
 import moa.classifiers.core.attributeclassobservers.DiscreteAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NominalAttributeClassObserver;
 import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
 import moa.classifiers.core.attributeclassobservers.NullAttributeClassObserver;
 import moa.classifiers.core.attributeclassobservers.NumericAttributeClassObserver;
import moa.classifiers.core.splitcriteria.InfoGainSplitCriterion;
 import moa.classifiers.core.splitcriteria.SplitCriterion;
 import moa.core.AutoExpandVector;
 import moa.core.DoubleVector;
 import moa.core.Measurement;
 import moa.core.StringUtils;
 import moa.options.ClassOption;
 import moa.options.FlagOption;
 import moa.options.FloatOption;
 import moa.options.IntOption;
 import moa.core.SizeOf;
 import moa.options.*;
 import weka.core.Instance;
 import weka.core.Utils;
 
 public class HoeffdingClassifier extends AbstractClassifier {
  @Override
  public String getPurposeString() {
    return "Hoeffding Tree or VFDT.";
  }
  public IntOption gracePeriodOption = new IntOption(
             "gracePeriod",
             'g',
             "The number of instances a leaf should observe between split attempts.",
             200, 0, Integer.MAX_VALUE);
  public FloatOption splitConfidenceOption = new FloatOption(
          "splitConfidence",
          'c',
          "The allowable error in split decision, values closer to 0 will take longer to decide.",
          0.0000001, 0.0, 1.0);
  public FloatOption tieThresholdOption = new FloatOption("tieThreshold",
          't', "Threshold below which a split will be forced to break ties.",
          0.05, 0.0, 1.0);
  public FlagOption binarySplitsOption = new FlagOption("binarySplits", 'b',
          "Only allow binary splits."); 
  public FlagOption removePoorAttsOption = new FlagOption("removePoorAtts",
          'r', "Disable poor attributes."); 
  public FlagOption noPrePruneOption = new FlagOption("noPrePrune", 'p',
          "Disable pre-pruning.");
  //***************************************************************************
  public MultiChoiceOption leafpredictionOption = new MultiChoiceOption(
     "leafprediction", 'l', "Leaf prediction to use.", new String[]{
         "MC", "NB", "NBAdaptive"}, new String[]{
         "Majority class",
         "Naive Bayes",
         "Naive Bayes Adaptive"}, 1);
  public IntOption nbThresholdOption = new IntOption(
          "nbThreshold",
          'q',
          "The number of instances a leaf should observe before permitting Naive Bayes.",
          0, 0, Integer.MAX_VALUE);  
    /**
    * Результат поиска в дереве
    */
  public static class FoundNode {
   public Node node;
   public SplitNode parent;
   public int parentBranch;
   public FoundNode(Node node, SplitNode parent, int parentBranch) {
     this.node = node;
     this.parent = parent;
     this.parentBranch = parentBranch;
   }
  }     
     /**
      * Узел дерева, аобстрактный класс
      */
  public static class Node extends AbstractMOAObject {
    protected DoubleVector observedClassDistribution;
    public Node(double[] classObservations) {
      this.observedClassDistribution = new DoubleVector(classObservations);
    }
    public int calcByteSize() {
      return (int) (SizeOf.sizeOf(this) + SizeOf.fullSizeOf(this.observedClassDistribution));
    }
    public int calcByteSizeIncludingSubtree() {
      return calcByteSize();
    }
    public boolean isLeaf() {
      return true;
    }
    /**
    * Поиск листа для экземпляра
    * @param inst
    * @param parent
    * @param parentBranch
    * @return 
    */
    public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent,int parentBranch) {
      return new FoundNode(this, parent, parentBranch);
    }
    public double[] getObservedClassDistribution() {
      return this.observedClassDistribution.getArrayCopy();
    }
    public double[] getClassVotes(Instance inst, HoeffdingClassifier ht) {
      return this.observedClassDistribution.getArrayCopy();
    }
    public boolean observedClassDistributionIsPure() {
      return this.observedClassDistribution.numNonZeroEntries() < 2;
    }
    public void describeSubtree(HoeffdingClassifier ht, StringBuilder out,int indent) {
      StringUtils.appendIndented(out, indent, "Leaf ");
      out.append(ht.getClassNameString());
      out.append(" = ");
      out.append(ht.getClassLabelString(this.observedClassDistribution.maxIndex()));
      out.append(" weights: ");
      this.observedClassDistribution.getSingleLineDescription(out,ht.treeRoot.observedClassDistribution.numValues());
      StringUtils.appendNewline(out);
    }
    public int subtreeDepth() {
      return 0;
    }
    public double calculatePromise() {
      double totalSeen = this.observedClassDistribution.sumOfValues();
      return totalSeen > 0.0 ? (totalSeen - this.observedClassDistribution.getValue(this.observedClassDistribution.maxIndex()))
                       : 0.0;
    }
    @Override
    public void getDescription(StringBuilder sb, int indent) {
      describeSubtree(null, sb, indent);
    }
  }
  /**
   * внутренний узел дерева с правилом разделения
   */
  public static class SplitNode extends Node {    
    protected InstanceConditionalTest splitTest; 
    protected AutoExpandVector<Node> children = new AutoExpandVector<Node>(); 
    @Override
    public int calcByteSize() {
      return super.calcByteSize()+ (int) (SizeOf.sizeOf(this.children) + SizeOf.fullSizeOf(this.splitTest));
  }
  @Override
  public int calcByteSizeIncludingSubtree() {
    int byteSize = calcByteSize();
    for (Node child : this.children) {
      if (child != null) {
        byteSize += child.calcByteSizeIncludingSubtree();
      }
    }
    return byteSize;
  } 
  public SplitNode(InstanceConditionalTest splitTest,double[] classObservations) {
    super(classObservations);
    this.splitTest = splitTest;
  }
  public int numChildren() {
    return this.children.size();
  }
  public void setChild(int index, Node child) {
    if ((this.splitTest.maxBranches() >= 0)
         && (index >= this.splitTest.maxBranches())) {
      throw new IndexOutOfBoundsException();
    }
    this.children.set(index, child);
  }
  public Node getChild(int index) {
       return this.children.get(index);
   }
  public int instanceChildIndex(Instance inst) {
      return this.splitTest.branchForInstance(inst);
  }
  @Override
  public boolean isLeaf() {
    return false;
  }
 
  @Override
  public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent,int parentBranch) {
    int childIndex = instanceChildIndex(inst);
    if (childIndex >= 0) {
      Node child = getChild(childIndex);
      if (child != null) {
        return child.filterInstanceToLeaf(inst, this, childIndex);
      }
      return new FoundNode(null, this, childIndex);
    }
    return new FoundNode(this, parent, parentBranch);
  } 
  @Override
  public void describeSubtree(HoeffdingClassifier ht, StringBuilder out,int indent) {
    for (int branch = 0; branch < numChildren(); branch++) {
      Node child = getChild(branch);
      if (child != null) {
        StringUtils.appendIndented(out, indent, "if ");
        out.append(this.splitTest.describeConditionForBranch(branch,ht.getModelContext()));
        out.append(": ");
        StringUtils.appendNewline(out);
        child.describeSubtree(ht, out, indent + 2);
      }
    }
  }
  @Override
  public int subtreeDepth() {
      int maxChildDepth = 0;
      for (Node child : this.children) {
          if (child != null) {
              int depth = child.subtreeDepth();
              if (depth > maxChildDepth) {
                  maxChildDepth = depth;
              }
          }
      }
      return maxChildDepth + 1;
  }
}
/**
 * лист дерева, абстрактный класс
 */
  public static abstract class LearningNode extends Node { 
    public LearningNode(double[] initialClassObservations) {
      super(initialClassObservations);
    }
    public abstract void learnFromInstance(Instance inst, HoeffdingClassifier ht);
  }
   /**
    * лист дерева, который не может делиться
    */
  public static class InactiveLearningNode extends LearningNode {
    public InactiveLearningNode(double[] initialClassObservations) {
      super(initialClassObservations);
    }
    @Override
    public void learnFromInstance(Instance inst, HoeffdingClassifier ht) {
      this.observedClassDistribution.addToValue((int) inst.classValue(),inst.weight());
    }
  }
  /**
   * лист дерева, который может делиться
   */
  public static class ActiveLearningNode extends LearningNode {
    protected double weightSeenAtLastSplitEvaluation;
    protected AutoExpandVector<AttributeClassObserver> attributeObservers = new AutoExpandVector<AttributeClassObserver>();
    public ActiveLearningNode(double[] initialClassObservations) {
      super(initialClassObservations);
      this.weightSeenAtLastSplitEvaluation = getWeightSeen();
    }
    @Override
    public int calcByteSize() {
      return super.calcByteSize()+ (int) (SizeOf.fullSizeOf(this.attributeObservers));
    }
    @Override
    public void learnFromInstance(Instance inst, HoeffdingClassifier ht) {
      this.observedClassDistribution.addToValue((int) inst.classValue(),inst.weight());
      for (int i = 0; i < inst.numAttributes() - 1; i++) {
        int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);       
        AttributeClassObserver obs = this.attributeObservers.get(i);
        if (obs == null) {
          obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
          this.attributeObservers.set(i, obs);
        }
        obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
      }
    }
    public double getWeightSeen() {
      return this.observedClassDistribution.sumOfValues();
    }
    public double getWeightSeenAtLastSplitEvaluation() {
        return this.weightSeenAtLastSplitEvaluation;
    }
    public void setWeightSeenAtLastSplitEvaluation(double weight) {
        this.weightSeenAtLastSplitEvaluation = weight;
    }
    /**
     * Возвращает предложения по дроблению дерева в виде массива AttributeSplitSuggestion
     * @param criterion
     * @param ht
     * @return 
     */
    public AttributeSplitSuggestion[] getBestSplitSuggestions(
      SplitCriterion criterion, HoeffdingClassifier ht) {
      List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();
      double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
      if (!ht.noPrePruneOption.isSet()) {
        // add null split as an option
        bestSuggestions.add(new AttributeSplitSuggestion(null,new double[0][], criterion.getMeritOfSplit(preSplitDist,new double[][]{preSplitDist})));
      }
      for (int i = 0; i < this.attributeObservers.size(); i++) {
        AttributeClassObserver obs = this.attributeObservers.get(i);
        if (obs != null) {
            AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion,
                    preSplitDist, i, ht.binarySplitsOption.isSet());
            if (bestSuggestion != null) {
                bestSuggestions.add(bestSuggestion);
            }
        }
      }
      return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
    }
 
    public void disableAttribute(int attIndex) {
      this.attributeObservers.set(attIndex, new NullAttributeClassObserver());
    }
  }
  /**
   * Текущий корень дерева
   */
  protected Node treeRoot;
  protected int decisionNodeCount;
  protected int activeLeafNodeCount;
  protected int inactiveLeafNodeCount;
  public int calcByteSize() {
    int size = (int) SizeOf.sizeOf(this);
    if (this.treeRoot != null) {
      size += this.treeRoot.calcByteSizeIncludingSubtree();
    }
    return size;
  }
  @Override
  public int measureByteSize() {
    return calcByteSize();
  }
  @Override
  public void resetLearningImpl() {
    this.treeRoot = null;
    this.decisionNodeCount = 0;
    this.activeLeafNodeCount = 0;
    this.inactiveLeafNodeCount = 0;
    if (this.leafpredictionOption.getChosenIndex()>0) { 
        this.removePoorAttsOption = null;
    }
  }
  /**
   * @param opt  0 -- Majority class; 1 -- Naive Bayes; 2 --Naive Bayes Adaptive
   */
  public void setLeafPredictionOption(int opt){
    this.leafpredictionOption.setChosenIndex(opt);     
  }
  @Override
  public void trainOnInstanceImpl(Instance inst) {    
     // если текущий узел пуст, считаем его листом        
    if (this.treeRoot == null) {
        this.treeRoot = newLearningNode();
        this.activeLeafNodeCount = 1;
    }
    //ищем узел для экземпляра
    FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
    Node leafNode = foundNode.node;
    //если узла не нашли, добавляем новый лист
    if (leafNode == null) {
      leafNode = newLearningNode();
      foundNode.parent.setChild(foundNode.parentBranch, leafNode);
      this.activeLeafNodeCount++;
    }
    // если узел -- лист обучаем его
    if (leafNode instanceof LearningNode) {
      LearningNode learningNode = (LearningNode) leafNode;
      learningNode.learnFromInstance(inst, this);
     // если лист может дробиться, проверяем соответствие количества элементов для дробления и пытаемся дробить
     if ((learningNode instanceof ActiveLearningNode)) {
        ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
        double weightSeen = activeLearningNode.getWeightSeen();
        if (weightSeen - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= this.gracePeriodOption.getValue()) {
          attemptToSplit(activeLearningNode, foundNode.parent,foundNode.parentBranch);
          activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
        }
      }
    }
  } 
  @Override
  public double[] getVotesForInstance(Instance inst) {
    if (this.treeRoot != null) {
      // ищем узел для экземпляра
      FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
      Node leafNode = foundNode.node;
      //если узел-- пуст, берем его родителя
      if (leafNode == null) {
        leafNode = foundNode.parent;
      }
      //  возвращаем результаты голосования
      return leafNode.getClassVotes(inst, this);
    }
    return new double[0];
  }
 
  @Override
  protected Measurement[] getModelMeasurementsImpl() {
      return new Measurement[]{
                  new Measurement("tree size (nodes)", this.decisionNodeCount
                  + this.activeLeafNodeCount + this.inactiveLeafNodeCount),
                  new Measurement("tree size (leaves)", this.activeLeafNodeCount
                  + this.inactiveLeafNodeCount),
                  new Measurement("active learning leaves",
                  this.activeLeafNodeCount),
                  new Measurement("tree depth", measureTreeDepth())};
  } 
  public int measureTreeDepth() {
    if (this.treeRoot != null) {
      return this.treeRoot.subtreeDepth();
    }
    return 0;
  }
  @Override
  public void getModelDescription(StringBuilder out, int indent) {
    this.treeRoot.describeSubtree(this, out, indent);
  }
  @Override
  public boolean isRandomizable() {
    return false;
  }
  public static double computeHoeffdingBound(double range, double confidence,double n) {
    return Math.sqrt(((range * range) * Math.log(1.0 / confidence))/ (2.0 * n));
  }
  //Procedure added for Hoeffding Adaptive Trees (ADWIN)
  protected SplitNode newSplitNode(InstanceConditionalTest splitTest,double[] classObservations) {
    return new SplitNode(splitTest, classObservations);
  }
  protected AttributeClassObserver newNominalClassObserver() {
    AttributeClassObserver nominalClassObserver = new NominalAttributeClassObserver();
    return (AttributeClassObserver) nominalClassObserver.copy();
  }
  protected AttributeClassObserver newNumericClassObserver() {
    AttributeClassObserver numericClassObserver = new GaussianNumericAttributeClassObserver();
    return (AttributeClassObserver) numericClassObserver.copy();
  }
 /**
  * Попытка разбить узел
  * @param node узел для разбивки
  * @param parent родитель узла
  * @param parentIndex 
  */
  protected void attemptToSplit(ActiveLearningNode node, SplitNode parent,int parentIndex) {
    // если в узле 2 и больше классов
    if (!node.observedClassDistributionIsPure()) {
      // получаем предложения по дроблению
      SplitCriterion splitCriterion =  new InfoGainSplitCriterion();
      AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
      //сортируем их
      Arrays.sort(bestSplitSuggestions);
      boolean shouldSplit = false;
      // если нет ни одного предложения -- не делим
      if (bestSplitSuggestions.length < 2) {
          shouldSplit = bestSplitSuggestions.length > 0;
      } else {
        //иначе проверяем условия с границей Хёфдинга
        double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
                                                      this.splitConfidenceOption.getValue(), node.getWeightSeen());
        //берем 2 лучших предложения
        AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
        AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];
        if ((bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound)
             || (hoeffdingBound < this.tieThresholdOption.getValue())) {
          shouldSplit = true;
        }
        //удаляем плохие атрибуты, если установили такую опцию
        if ((this.removePoorAttsOption != null)&& this.removePoorAttsOption.isSet()) {
          Set<Integer> poorAtts = new HashSet<Integer>();
          // scan 1 - add any poor to set
          for (int i = 0; i < bestSplitSuggestions.length; i++) {
            if (bestSplitSuggestions[i].splitTest != null) {
              int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
              if (splitAtts.length == 1) {               
                if (bestSuggestion.merit - bestSplitSuggestions[i].merit > hoeffdingBound) {
                  poorAtts.add(new Integer(splitAtts[0]));
                }
              }
            }
          }
          // scan 2 - remove good ones from set          
          for (int i = 0; i < bestSplitSuggestions.length; i++) {
            if (bestSplitSuggestions[i].splitTest != null) {
              int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
              if (splitAtts.length == 1) {
                if (bestSuggestion.merit- bestSplitSuggestions[i].merit < hoeffdingBound) {
                  poorAtts.remove(new Integer(splitAtts[0]));
                }
              }
            }
          }
          // удаляем те атрибуты, которые не удовлетворяют "условию хёфдинга", в целях экономии памяти
          for (int poorAtt : poorAtts) {
              node.disableAttribute(poorAtt);
          }
        }
      }
      if (shouldSplit) {
        AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];
        // если выиграл вариант "не делить" деактивируем узел, он больше не будет делиться
        if (splitDecision.splitTest == null) {
          // preprune - null wins
          deactivateLearningNode(node, parent, parentIndex);
        } else {
          // иначе создаем новый узел с правилом, и его потомками делаем новые узлы.
          SplitNode newSplit = newSplitNode(splitDecision.splitTest,node.getObservedClassDistribution());
          for (int i = 0; i < splitDecision.numSplits(); i++) {
            Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));
            newSplit.setChild(i, newChild);
          }
          this.activeLeafNodeCount--;
          this.decisionNodeCount++;
          this.activeLeafNodeCount += splitDecision.numSplits();
          if (parent == null) {
            this.treeRoot = newSplit;
          } else {
            parent.setChild(parentIndex, newSplit);
          }
        }
      }
    }
  }
  protected void deactivateLearningNode(ActiveLearningNode toDeactivate,SplitNode parent, int parentBranch) {
    Node newLeaf = new InactiveLearningNode(toDeactivate.getObservedClassDistribution());
    if (parent == null) {
      this.treeRoot = newLeaf;
    } else {
      parent.setChild(parentBranch, newLeaf);
    }
    this.activeLeafNodeCount--;
    this.inactiveLeafNodeCount++;
  }
 /**
  * Лист, использующий Байесовское распределение, чтобы предсказать значение
  */
  public static class LearningNodeNB extends ActiveLearningNode {
    public LearningNodeNB(double[] initialClassObservations) {
      super(initialClassObservations);
    }
    @Override
    public double[] getClassVotes(Instance inst, HoeffdingClassifier ht) {
      if (getWeightSeen() >= ht.nbThresholdOption.getValue()) {
        return NaiveBayes.doNaiveBayesPrediction(inst,this.observedClassDistribution,this.attributeObservers);
      }
      return super.getClassVotes(inst, ht);
    } 
   @Override
   public void disableAttribute(int attIndex) {
   // should not disable poor atts - they are used in NB calc
   }
  }
  /**
   * Адаптивный лист, применяющий MC если он получает лучшие веса или NB если он получает лучшие веса
   */
  public static class LearningNodeNBAdaptive extends LearningNodeNB {
    protected double mcCorrectWeight = 0.0;
    protected double nbCorrectWeight = 0.0;
    public LearningNodeNBAdaptive(double[] initialClassObservations) {
      super(initialClassObservations);
    }
    @Override
    public void learnFromInstance(Instance inst, HoeffdingClassifier ht) {
      int trueClass = (int) inst.classValue();
      if (this.observedClassDistribution.maxIndex() == trueClass) {
        this.mcCorrectWeight += inst.weight();
      }
      if (Utils.maxIndex(NaiveBayes.doNaiveBayesPrediction(inst,
                                                           this.observedClassDistribution, 
                                                           this.attributeObservers)) 
                         == trueClass) {
        this.nbCorrectWeight += inst.weight();
      }
      super.learnFromInstance(inst, ht);
    }
    @Override
    public double[] getClassVotes(Instance inst, HoeffdingClassifier ht) {
      if (this.mcCorrectWeight > this.nbCorrectWeight) {
        return this.observedClassDistribution.getArrayCopy();     
      }
      return NaiveBayes.doNaiveBayesPrediction(inst,this.observedClassDistribution, this.attributeObservers);
    }
  } 
  protected LearningNode newLearningNode() {
      return newLearningNode(new double[0]);
  }
  protected LearningNode newLearningNode(double[] initialClassObservations) {
    LearningNode ret;
    int predictionOption = this.leafpredictionOption.getChosenIndex();
    if (predictionOption == 0) { //MC
        ret = new ActiveLearningNode(initialClassObservations);
    } else if (predictionOption == 1) { //NB
        ret = new LearningNodeNB(initialClassObservations);
    } else { //NBAdaptive
        ret = new LearningNodeNBAdaptive(initialClassObservations);
    }
    return ret;
  }
}