 package com.mycompany.bachelorwork.BachelorClassifiers; 
 import weka.core.Instance;
 import weka.core.Utils;
 
 import java.io.File;
 import java.io.FileOutputStream;
 import java.io.PrintStream;
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
 import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
 import moa.classifiers.core.attributeclassobservers.NumericAttributeClassObserver;
import moa.classifiers.core.splitcriteria.InfoGainSplitCriterion;
 import moa.classifiers.core.splitcriteria.SplitCriterion;
 import moa.core.AutoExpandVector;
 import moa.core.DoubleVector;
 import moa.core.Measurement;
 import moa.core.SizeOf;
 import moa.core.StringUtils;
 import moa.options.*; 
public class HoeffdingOptionClassifier extends AbstractClassifier {
  @Override
  public String getPurposeString() {
    return "Hoeffding Option Tree: single tree that represents multiple trees.";
  }
  public IntOption maxOptionPathsOption = new IntOption(
    "maxOptionPaths",
    'o', 
    "Maximum number of option paths per node.",  
    5, 1,Integer.MAX_VALUE);
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
  public FloatOption secondarySplitConfidenceOption = new FloatOption(
    "secondarySplitConfidence",
    'w',
    "The allowable error in secondary split decisions, values closer to 0 will take longer to decide.",
    0.1, 0.0, 1.0);
  public FloatOption tieThresholdOption = new FloatOption(
    "tieThreshold",
    't', "Threshold below which a split will be forced to break ties.",
    0.05, 0.0, 1.0);
  public FlagOption binarySplitsOption = new FlagOption(
    "binarySplits", 
    'b',
    "Only allow binary splits."); 
  public FlagOption removePoorAttsOption = new FlagOption(
    "removePoorAtts",
    'r', 
    "Disable poor attributes.");
  public FlagOption noPrePruneOption = new FlagOption(
    "noPrePrune", 
    'p',
    "Disable pre-pruning.");
 //*****************************************************************************
  public MultiChoiceOption leafpredictionOption = new MultiChoiceOption(
    "leafprediction", 
    'l', 
    "Leaf prediction to use.",
    new String[]{"MC", "NB", "NBAdaptive"}, 
    new String[]{"Majority class","Naive Bayes","Naive Bayes Adaptive"}, 
    2);
  public IntOption nbThresholdOption = new IntOption(
    "nbThreshold",
    'q',
    "The number of instances a leaf should observe before permitting Naive Bayes.",
    0, 0, Integer.MAX_VALUE);
  /**
   * Класс для представления результата поиска
   */
  public static class FoundNode {
    public Node node;
    public SplitNode parent;
    public int parentBranch; // set to -999 for option leaves
    public FoundNode(Node node, SplitNode parent, int parentBranch) {
      this.node = node;
      this.parent = parent;
      this.parentBranch = parentBranch;
    }
  }
  /**
   * Абстрактный класс узла дерева
   */ 
  public static class Node extends AbstractMOAObject {     
    /**
     * Распределение классов
     * 
     */
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
     * Поиск листа для экземпляра inst. Запускает алгоритм для поиска многих листьев
     * @param inst
     * @param parent
     * @param parentBranch
     * @param updateSplitterCounts
     * @return 
     */
    public FoundNode[] filterInstanceToLeaves(Instance inst,SplitNode parent, int parentBranch, boolean updateSplitterCounts) { 
     List<FoundNode> nodes = new LinkedList<FoundNode>();
     filterInstanceToLeaves(inst, parent, parentBranch, nodes,updateSplitterCounts);
     return nodes.toArray(new FoundNode[nodes.size()]);
    }
    /**
     * Поиск листьев для экземпляра. добавляет текущий узел  в возращаемую колекцию
     * @param inst
     * @param splitparent
     * @param parentBranch
     * @param foundNodes
     * @param updateSplitterCounts 
     */
    public void filterInstanceToLeaves(Instance inst,
                                       SplitNode splitparent, int parentBranch,
                                       List<FoundNode> foundNodes, boolean updateSplitterCounts) {
      foundNodes.add(new FoundNode(this, splitparent, parentBranch));
    }
    /**
     * Возвращает распределение классов узла
     * @return 
     */
    public double[] getObservedClassDistribution() {
        return this.observedClassDistribution.getArrayCopy();
    }
    /**
     * Возвращает голоса для  выбора класса экземпляра в соответствии с деревом.
     * @param inst
     * @param ht
     * @return 
     */ 
    public double[] getClassVotes(Instance inst, HoeffdingOptionClassifier ht) {
      double[] dist = this.observedClassDistribution.getArrayCopy();
      double distSum = Utils.sum(dist);
      if (distSum > 0.0) {
        Utils.normalize(dist, distSum);
      }
      return dist;
    }
    /**
     * проверяет является ли распределение классов слишком бедным, не представительным
     * @return 
     */
    public boolean observedClassDistributionIsPure() {
        return this.observedClassDistribution.numNonZeroEntries() < 2;
    }
    /**
     * Описание поддерева в виде строки.
     * @param ht
     * @param out
     * @param indent 
     */ 
    public void describeSubtree(HoeffdingOptionClassifier ht, StringBuilder out,int indent) {
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
      return totalSeen > 0.0 ? 
        (totalSeen - this.observedClassDistribution.getValue(this.observedClassDistribution.maxIndex()))
                             : 0.0;
    }
 
    public void getDescription(StringBuilder sb, int indent) {
        describeSubtree(null, sb, indent);
    }
  }
  /**
   * Класс внутреннего узла дерева.
   */ 
  public static class SplitNode extends Node {  
    protected InstanceConditionalTest splitTest; // критерий разбиения в данном узле 
    protected SplitNode parent;  // родитель узла
    protected Node nextOption; //альтернативная ветка 
    protected int optionCount; // set to -999 for optional splits
    protected AutoExpandVector<Node> children = new AutoExpandVector<Node>();  // дети узла
    @Override
    public int calcByteSize() {
      return super.calcByteSize()
             + (int) (SizeOf.sizeOf(this.children) + SizeOf.fullSizeOf(this.splitTest));
    }
    @Override
    public int calcByteSizeIncludingSubtree() {
      int byteSize = calcByteSize();
      for (Node child : this.children) {
        if (child != null) {
          byteSize += child.calcByteSizeIncludingSubtree();
        }
      }
      if (this.nextOption != null) {
        byteSize += this.nextOption.calcByteSizeIncludingSubtree();
      }
      return byteSize;
    }
    /**
     * Конструктор внутреннего узла
     * @param splitTest
     * @param classObservations 
     */ 
    public SplitNode(InstanceConditionalTest splitTest,
            double[] classObservations) {
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
    /**
    * Нахождение нужной ветки экземпляра
    * @param inst
    * @return 
    */
    public int instanceChildIndex(Instance inst) {
      return this.splitTest.branchForInstance(inst);
    }

    @Override
    public boolean isLeaf() {
      return false;
    }
    /**
     * Поиск листов для экземпляра
     * @param inst
     * @param myparent
     * @param parentBranch
     * @param foundNodes
     * @param updateSplitterCounts 
     */
    @Override
    public void filterInstanceToLeaves(Instance inst, SplitNode myparent,
                                        int parentBranch, List<FoundNode> foundNodes,
                                        boolean updateSplitterCounts) {
      // добавляем узел в распределение, если нужно
      if (updateSplitterCounts) {
        this.observedClassDistribution.addToValue((int) inst.classValue(), inst.weight());
      }   
      int childIndex = instanceChildIndex(inst); // находим нужную ветку для экземпляра
      if (childIndex >= 0) {  // если ветка существует
        Node child = getChild(childIndex);//находим ребенка по этой ветке
        if (child != null) { // если он не пуст, запускаем алгоритм дальше от ребенка
          child.filterInstanceToLeaves(inst, this, childIndex,foundNodes, updateSplitterCounts);
        } else {
          foundNodes.add(new FoundNode(null, this, childIndex)); // если пуст, мы пришли куда нужно
        }
      }
      if (this.nextOption != null) {// если есть альтернативная ветка, запускаем метод от неё
        this.nextOption.filterInstanceToLeaves(inst, this, -999, foundNodes, updateSplitterCounts);
      }
    }
    /**
     * описание поддерева
     * @param ht
     * @param out
     * @param indent 
     */
    @Override
    public void describeSubtree(HoeffdingOptionClassifier ht, StringBuilder out,int indent) {
      for (int branch = 0; branch < numChildren(); branch++) {
        Node child = getChild(branch);
        if (child != null) {
          StringUtils.appendIndented(out, indent, "if ");
          out.append(this.splitTest.describeConditionForBranch(branch,ht.getModelContext()));
          out.append(": ");
          out.append("** option count = " + this.optionCount);
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
    /**
     * Метод, считающий полезность разбиения с использованием splitCriterion
     * @param splitCriterion
     * @param preDist
     * @return 
     */ 
    public double computeMeritOfExistingSplit(SplitCriterion splitCriterion, double[] preDist) {
      double[][] postDists = new double[this.children.size()][];
      for (int i = 0; i < this.children.size(); i++) {
        if (this.children.get(i)!=null){
          postDists[i] = this.children.get(i).getObservedClassDistribution();
        }
        else 
        {
          postDists[i] =new  double[0];
        }    
      }

      return splitCriterion.getMeritOfSplit(preDist, postDists);
    }
    /**
     * Обновляет счетчик альтернативных узлов.
     * @param source
     * @param hot 
     */ 
    public void updateOptionCount(SplitNode source, HoeffdingOptionClassifier hot) {
      // если узел - альтернативный, идем выше.
      if (this.optionCount == -999) {
          this.parent.updateOptionCount(source, hot);
      } else {
        int maxChildCount = -999;
        // берем текущий узел
        SplitNode curr = this;
        // пока он не пуст ищем максимальное число альтернативных узлов в детях и альтернативных
        while (curr != null) {
          //проходим по детям и ищем узел с большим количеством альтернативных узлов.
          for (Node child : curr.children) {
            if (child instanceof SplitNode) {
              SplitNode splitChild = (SplitNode) child;
              if (splitChild.optionCount > maxChildCount) {
                maxChildCount = splitChild.optionCount;
              }
            }
          }
          // если есть альтернативный узел, и он именно узел, то переходим к нему
          if ((curr.nextOption != null)&& (curr.nextOption instanceof SplitNode)) {
            curr = (SplitNode) curr.nextOption;
          } else {
            // иначе обнуляем текущий узел( выходим из цикла)
            curr = null;
          }
        }
        // если наибольшее число альтернативных больше текущего числа альтернативных
        if (maxChildCount > this.optionCount) { // currently only works
          // one
          // way - adding, not
          // removing
          // запоминаем разницу
          int delta = maxChildCount - this.optionCount;
          //обновляем инфу в узле
          this.optionCount = maxChildCount;
          //если альтернатив слишком много, убиваем узлы
          if (this.optionCount >= hot.maxOptionPathsOption.getValue()) {
            killOptionLeaf(hot);
          }
          curr = this;
          // пока текущий узел не пуст
          while (curr != null) {
            // проходим по детям 
            for (Node child : curr.children) {
              // если ребенок -- узел, обновляем OptionCount в нем  на величину delta
              if (child instanceof SplitNode) {
                SplitNode splitChild = (SplitNode) child;
                if (splitChild != source) {
                  splitChild.updateOptionCountBelow(delta,hot);
                }
              }
            }
            // Если есть альтернативы, переходим к ним для следующей итерации цикла
            if ((curr.nextOption != null)&& (curr.nextOption instanceof SplitNode)) {
              curr = (SplitNode) curr.nextOption;
            } else {
              // если нет выходим из цикла, обнуляем curr
              curr = null;
            }
          }
          // если у текущего узла есть родитель, обновляем optionCount в нем
          if (this.parent != null) {
            this.parent.updateOptionCount(this, hot);
          }
        }
      }
    }
    /**
     * Обновление OptionCount в потомках на величину delta
     * @param delta
     * @param hot 
     */ 
    public void updateOptionCountBelow(int delta, HoeffdingOptionClassifier hot) {
      // если узел не альтернативный
      if (this.optionCount != -999) {
        // обновляем величину, и если итоговое количество  альтернатив слишком велико, удаляем ветки
        this.optionCount += delta;
        if (this.optionCount >= hot.maxOptionPathsOption.getValue()) {
          killOptionLeaf(hot);
        }
      }
      // для всех детей, если они узлы, обновляем в них OptionCount
      for (Node child : this.children) {
        if (child instanceof SplitNode) {
          SplitNode splitChild = (SplitNode) child;
          splitChild.updateOptionCountBelow(delta, hot);
        }
      }
      // для всех альтернативных ветвей, обновляем OptionCount
      if (this.nextOption instanceof SplitNode) {
        ((SplitNode) this.nextOption).updateOptionCountBelow(delta, hot);
      }
    }
    /**
     * Метод удаления избыточных альтернативных веток
     * @param hot 
     */
    private void killOptionLeaf(HoeffdingOptionClassifier hot) {
      if (this.nextOption instanceof SplitNode) {
        ((SplitNode) this.nextOption).killOptionLeaf(hot);
      } else if (this.nextOption instanceof ActiveLearningNode) {
        this.nextOption = null;
        hot.activeLeafNodeCount--;
      } else if (this.nextOption instanceof InactiveLearningNode) {
        this.nextOption = null;
        hot.inactiveLeafNodeCount--;
      }
    }
    /**
     * для альтернативной ветки выясняем количество альтернативных веток.
     * @return 
     */
    public int getHeadOptionCount() {
      SplitNode sn = this;
      while (sn.optionCount == -999) {
        sn = sn.parent;
      }
      return sn.optionCount;
    }
  }
  /**
  * Абстрактный класс листа
  */
  public static abstract class LearningNode extends Node {
    public LearningNode(double[] initialClassObservations) {
      super(initialClassObservations);
    }
    public abstract void learnFromInstance(Instance inst,HoeffdingOptionClassifier ht);
  }
  /**
   * узел, не могущий дробиться
   */
  public static class InactiveLearningNode extends LearningNode {
    public InactiveLearningNode(double[] initialClassObservations) {
      super(initialClassObservations);
    }
    /**
    * учесть экземпляр в статистику
    * @param inst
    * @param ht 
    */
    @Override
    public void learnFromInstance(Instance inst, HoeffdingOptionClassifier ht) {
        this.observedClassDistribution.addToValue((int) inst.classValue(), inst.weight());
    }
  }
  /**
   * Узел, могущий дробиться
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
    /**
     * метод учитывающий экземпляр в статистику
     * @param inst
     * @param ht 
     */
    @Override
    public void learnFromInstance(Instance inst, HoeffdingOptionClassifier ht) {
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
     * Возвращает Предложения по разбиению по критерию
     * @param criterion
     * @param ht
     * @return 
     */
    public AttributeSplitSuggestion[] getBestSplitSuggestions(SplitCriterion criterion, HoeffdingOptionClassifier ht) {
      List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();
      double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
      // добавляем опцию "не делиться", если можно
      if (!ht.noPrePruneOption.isSet()) {
        // add null split as an option
        bestSuggestions.add(new AttributeSplitSuggestion(null,
                                                         new double[0][], 
                                                         criterion.getMeritOfSplit(preSplitDist,new double[][]{preSplitDist})
                                                        )
                            );
      }
      // для каждого атрибута, добавляем полученное предложение, если оно не пусто
      for (int i = 0; i < this.attributeObservers.size(); i++) {
        AttributeClassObserver obs = this.attributeObservers.get(i);
        if (obs != null) {            
          AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion,
                                                                                        preSplitDist, 
                                                                                        i, 
                                                                                        ht.binarySplitsOption.isSet()
                                                                                        );
          if (bestSuggestion != null) {
              bestSuggestions.add(bestSuggestion);
          }
        }
      }
      return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
    }
    public void disableAttribute(int attIndex) {
      this.attributeObservers.set(attIndex,new NullAttributeClassObserver());
    }
  }
  protected Node treeRoot; // корень дерева
 
  protected int decisionNodeCount; 
 
  protected int activeLeafNodeCount;
 
  protected int inactiveLeafNodeCount;

  protected int maxPredictionPaths;
 
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
    this.maxPredictionPaths = 0;
    if (this.leafpredictionOption.getChosenIndex() > 0) {
        this.removePoorAttsOption = null;
    }
  }
  /**
  * метод обучения на экземпляре
  * 
  * @param inst 
  */
  public int cnt=0;
  @Override
  public void trainOnInstanceImpl(Instance inst) {
    cnt++;
    // если корень пуст, просто создаем новый лист
    if (this.treeRoot == null) {
      this.treeRoot = newLearningNode();
      this.activeLeafNodeCount = 1;
    }   
    // получем список возможных листов
    FoundNode[] foundNodes = this.treeRoot.filterInstanceToLeaves(inst,null, -1, true);
    // для каждого листа  
    for (FoundNode foundNode : foundNodes) {
      // option leaves will have a parentBranch of -999
      // option splits will have an option count of -999
      // создаем лист для дерева
      Node leafNode = foundNode.node;
      if (leafNode == null) {
        leafNode = newLearningNode();
        foundNode.parent.setChild(foundNode.parentBranch, leafNode);
        this.activeLeafNodeCount++;
      }
      // если найденый лист -- лист дерева, обновляем статистику и пытаемся делиться, если лист активный
      if (leafNode instanceof LearningNode) {
        LearningNode learningNode = (LearningNode) leafNode;
        learningNode.learnFromInstance(inst, this);
        if (learningNode instanceof ActiveLearningNode) {
          ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
          double weightSeen = activeLearningNode.getWeightSeen();         
          if (weightSeen - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= this.gracePeriodOption.getValue()) {
             attemptToSplit(activeLearningNode, foundNode.parent,foundNode.parentBranch);
             activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
          }
        }
      }
    }      
  }
  public int flag=0;
  /**
   * получает голоса для экземпляра
   * @param inst
   * @return 
   */
  @Override
  public double[] getVotesForInstance(Instance inst) {
      if (this.treeRoot != null) {
        // ищем листья
          FoundNode[] foundNodes = this.treeRoot.filterInstanceToLeaves(inst,
                  null, -1, false);
          DoubleVector result = new DoubleVector();
          int predictionPaths = 0;
          // для каждого листа ищем голоса
          for (FoundNode foundNode : foundNodes) {
              if (foundNode.parentBranch != -999) {
                  Node leafNode = foundNode.node;
                  if (leafNode == null) {
                      leafNode = foundNode.parent;
                  }
                  double[] dist = leafNode.getClassVotes(inst, this);
                  //Albert: changed for weights
                  //double distSum = Utils.sum(dist);
                  //if (distSum > 0.0) {
                  //  Utils.normalize(dist, distSum);
                  //}
                  result.addValues(dist);
                  predictionPaths++;
              }
          }
          // обновляем максимальное число возможных вариантов предсказания для дерева
          if (predictionPaths > this.maxPredictionPaths) {
              this.maxPredictionPaths++;
          }
          return result.getArrayRef();
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
                  new Measurement("tree depth", measureTreeDepth()),               
                  new Measurement("maximum prediction paths used",
                  this.maxPredictionPaths)};
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
 
  public static double computeHoeffdingBound(double range, double confidence,
          double n) {
      return Math.sqrt(((range * range) * Math.log(1.0 / confidence))
              / (2.0 * n));
  }
 
  protected AttributeClassObserver newNominalClassObserver() {
      AttributeClassObserver nominalClassObserver = new NominalAttributeClassObserver();
      return (AttributeClassObserver) nominalClassObserver.copy();
  }
 
  protected AttributeClassObserver newNumericClassObserver() {
      AttributeClassObserver numericClassObserver =new GaussianNumericAttributeClassObserver();
      return (AttributeClassObserver) numericClassObserver.copy();
  }
  /**
   * Попытка разбить узел
   * @param node
   * @param parent
   * @param parentIndex 
   */
  protected void attemptToSplit(ActiveLearningNode node, SplitNode parent,int parentIndex) {
    // если распределение классов информативно
    if (!node.observedClassDistributionIsPure()) {
      //получаем критерий разбиения
      SplitCriterion splitCriterion = new InfoGainSplitCriterion();
      // получаем предложения по разбиению
      AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
      //сортируем их
      Arrays.sort(bestSplitSuggestions);
      boolean shouldSplit = false;
      // если лист не альтернативный
      if (parentIndex != -999) {
        if (bestSplitSuggestions.length < 2) {
          shouldSplit = bestSplitSuggestions.length > 0;
        } else {
          // считаем границу Хёфдинга
          double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
                                                        this.splitConfidenceOption.getValue(), 
                                                        node.getWeightSeen()
                                                        );
          // Выбираем два лучшех предложения по разбиению
          AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
          AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];
          // сравниваем разность полезностей с границей хефдинга, если ок, решаем делить
          if ((bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound)
              || (hoeffdingBound < this.tieThresholdOption.getValue())) {
            shouldSplit = true;
          }
          // Если должны удалять плохие атрибуты -- удаляем
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
            for (int poorAtt : poorAtts) {
              node.disableAttribute(poorAtt);
            }
          }
        }
        // Если лист альтернативный  
      } else if (bestSplitSuggestions.length > 0) {
        // считаем границу хефдига
        double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
        this.secondarySplitConfidenceOption.getValue(), node.getWeightSeen());
        // считаем лучшее предложение 
        AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
        // in option case, scan back through existing options to
        // find best      
        SplitNode current = parent;
        double bestPreviousMerit = Double.NEGATIVE_INFINITY;
        double[] preDist = node.getObservedClassDistribution();
        // проходим по предкам-альтернативам, в поисках лучшей прошлой полезности разбиения
        while (true) {
          double merit = current.computeMeritOfExistingSplit(
                  splitCriterion, preDist);
          if (merit > bestPreviousMerit) {
              bestPreviousMerit = merit;
          }
          if (current.optionCount != -999) {
              break;
          }
          current = current.parent;
        }
        // разность полезности текущего и прошлого разбиений сравниваем с границей и если ок, делим
        if (bestSuggestion.merit - bestPreviousMerit > hoeffdingBound) {
          shouldSplit = true;
        }
      }
      // Делим, если должны
      if (shouldSplit) {
        AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];
        //Если победил вариант не делить, то деактивируем лист( не альтернативный)
        if (splitDecision.splitTest == null) {
          // preprune - null wins          
          if (parentIndex != -999) {
            deactivateLearningNode(node, parent, parentIndex);
          }
        } else {
          SplitNode newSplit = new SplitNode(splitDecision.splitTest,node.getObservedClassDistribution());
          newSplit.parent = parent;
          // add option procedure
          SplitNode optionHead = parent;
          if (parent != null) {
            // ищем начало альтернативной ветки
            while (optionHead.optionCount == -999) {
              optionHead = optionHead.parent;
            }
          }
          // если текущий лист -- альтернативный и имеет родителя
          if ((parentIndex == -999) && (parent != null)) {
            // adding a new option                
            newSplit.optionCount = -999;
            // обновляем  OptionCunt у потомков начала ветки
            optionHead.updateOptionCountBelow(1, this);
            if (optionHead.parent != null) {
              // Обновляем OptionCount у родителя начала ветки
              optionHead.parent.updateOptionCount(optionHead,this);
            }        
          } else {            
            // adding a regular leaf
            if (optionHead == null) {
              newSplit.optionCount = 1;
            } else {
              newSplit.optionCount = optionHead.optionCount;
            }
          }
          int numOptions = 1;
          if (optionHead != null) {
            numOptions = optionHead.optionCount;
          }
          // альтернативных веток не слишком много, сохраняем прошлый лист как альтернативный
          if (numOptions < this.maxOptionPathsOption.getValue()) {
            newSplit.nextOption = node; // preserve leaf
            // disable attribute just used
            int[] splitAtts = splitDecision.splitTest.getAttsTestDependsOn();
            for (int i : splitAtts) {
              node.disableAttribute(i);
            }
          } else {
            // иначе не сохраняем
            this.activeLeafNodeCount--;
          }
          for (int i = 0; i < splitDecision.numSplits(); i++) {
            Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));      
            newSplit.setChild(i, newChild); 
          }
          flag=1;
          this.decisionNodeCount++;
          this.activeLeafNodeCount += splitDecision.numSplits();
          if (parent == null) {
            this.treeRoot = newSplit;
          } else {
            if (parentIndex != -999) {
              parent.setChild(parentIndex, newSplit);   
            } else {
              parent.nextOption = newSplit;
            }
          }
        }
      }
    }
  }
  /**
  * Функция добавления разбиения в таблицу альтернатиных
  * @param bestSuggestion
  * @param parent 
  */
  protected void deactivateLearningNode(ActiveLearningNode toDeactivate,SplitNode parent, int parentBranch) {
    Node newLeaf = new InactiveLearningNode(toDeactivate.getObservedClassDistribution());
    if (parent == null) {
      this.treeRoot = newLeaf;
    } else {
      if (parentBranch != -999) {
        parent.setChild(parentBranch, newLeaf);
      } else {
        parent.nextOption = newLeaf;
      }
    }
    this.activeLeafNodeCount--;
    this.inactiveLeafNodeCount++;
  }
  public static class LearningNodeNB extends ActiveLearningNode {
    public LearningNodeNB(double[] initialClassObservations) {
      super(initialClassObservations);
    }
    @Override
    public double[] getClassVotes(Instance inst, HoeffdingOptionClassifier hot) {
      if (getWeightSeen() >=  hot.nbThresholdOption.getValue()) {
        return NaiveBayes.doNaiveBayesPrediction(inst,this.observedClassDistribution,this.attributeObservers);
      }
      return super.getClassVotes(inst, hot);
    }
    @Override
    public void disableAttribute(int attIndex) {
      //should not disable poor atts - they are used in NB calc
    }
  }
  public static class LearningNodeNBAdaptive extends LearningNodeNB {
    protected double mcCorrectWeight = 0.0;
    protected double nbCorrectWeight = 0.0;
    public LearningNodeNBAdaptive(double[] initialClassObservations) {
      super(initialClassObservations);
    }
    @Override
    public void learnFromInstance(Instance inst, HoeffdingOptionClassifier hot) {
      int trueClass = (int) inst.classValue();
      if (this.observedClassDistribution.maxIndex() == trueClass) {
        this.mcCorrectWeight += inst.weight();
      }
      if (Utils.maxIndex(NaiveBayes.doNaiveBayesPrediction(inst,
                                                           this.observedClassDistribution, 
                                                           this.attributeObservers)
                        ) == trueClass) {
        this.nbCorrectWeight += inst.weight();
      }
      super.learnFromInstance(inst, hot);
    }
 
    @Override
    public double[] getClassVotes(Instance inst, HoeffdingOptionClassifier ht) {
        if (this.mcCorrectWeight > this.nbCorrectWeight) {
            return this.observedClassDistribution.getArrayCopy();
        }
        return NaiveBayes.doNaiveBayesPrediction(inst,
                this.observedClassDistribution, this.attributeObservers);
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
