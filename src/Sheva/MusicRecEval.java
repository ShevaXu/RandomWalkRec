package Sheva;

import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.*;
import org.apache.mahout.cf.taste.impl.model.GenericBooleanPrefDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.*;
import org.apache.mahout.cf.taste.impl.recommender.*;
import org.apache.mahout.cf.taste.impl.recommender.svd.*;
import org.apache.mahout.cf.taste.impl.similarity.*;
import org.apache.mahout.cf.taste.model.*;
import org.apache.mahout.cf.taste.neighborhood.*;
import org.apache.mahout.cf.taste.recommender.*;
import org.apache.mahout.cf.taste.similarity.*;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.*;

import java.io.*;
import java.util.*;

public class MusicRecEval {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		// DataModel model = new FileDataModel(new File("intro.csv"));

		DataModel model = new GenericBooleanPrefDataModel(
				GenericBooleanPrefDataModel.toDataMap(new FileDataModel(
						new File("occf-matrix"))));
		
		System.out.println("total users: " + model.getNumUsers());

		// RandomUtils.useTestSeed();
		OccfEvaluator eval = new OccfEvaluator();

		DataModelBuilder modelBuilder = new DataModelBuilder() {
			@Override
			public DataModel buildDataModel(
					FastByIDMap<PreferenceArray> trainingData) {
				return new GenericBooleanPrefDataModel(
						GenericBooleanPrefDataModel.toDataMap(trainingData));
			}
		};
		
		RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
			@Override
			public Recommender buildRecommender(DataModel model) throws TasteException {
//				return new SVDRecommender(model, new ALSWRFactorizer(model, 20, 0.5, 20));
//				String file = "aid-sids";
//			  	return new RandomWalkRecommender(model, file);
//				return new RandomWalkRecommender(model, null);
				UserSimilarity similarity = new CachingUserSimilarity(new TanimotoCoefficientSimilarity(model), model);
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(50, similarity, model);
				return new GenericBooleanPrefUserBasedRecommender(model, neighborhood, similarity);
//				ItemSimilarity similarity = new CachingItemSimilarity(new TanimotoCoefficientSimilarity(model), model);
//				return new GenericBooleanPrefItemBasedRecommender(model, similarity);
			}
		};
		
		FileOutputStream fos = new FileOutputStream("result.txt");
		OutputStreamWriter osw = new OutputStreamWriter(fos);
		BufferedWriter bw = new BufferedWriter(osw);
		MyIRStatic irs;
		double ndcg = 0.0;
		double stddev = 0.0;
		int runTimes = 20;
		for(int j = 5; j <= 40; j*=2) {
			bw.write("NDCG@" + j + " result: \n");
			for (int i = 0; i < runTimes; i++) {				
				irs = (MyIRStatic) eval.evaluate(recommenderBuilder, modelBuilder, model, null, 
						j,
						0.8,
						0.2
						);
				ndcg += irs.getNormalizedDiscountedCumulativeGain();
				stddev += irs.getStdDev();
			}
			ndcg /= (double)runTimes;
			stddev /= (double)runTimes;
			bw.write("Average ndcg: " + ndcg + "Average stddev: " + stddev + " \n");
		}			
		bw.close();
		
//		RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
//			@Override
//			public Recommender buildRecommender(DataModel model) throws TasteException {
////				return new SVDRecommender(model, new ALSWRFactorizer(model, 20, 0.5, 20));
////				UserSimilarity similarity = new CachingUserSimilarity(new TanimotoCoefficientSimilarity(model), model);
////				UserNeighborhood neighborhood = new NearestNUserNeighborhood(20, similarity, model);
////				return new GenericBooleanPrefUserBasedRecommender(model, neighborhood, similarity);
////				ItemSimilarity similarity = new CachingItemSimilarity(new TanimotoCoefficientSimilarity(model), model);
////				return new GenericBooleanPrefItemBasedRecommender(model, similarity);
////				return new RandomRecommender(model);
//				String file = "aid-sids";
//			  	return new RandomWalkRecommender(model, file);
//			}
//		};
//		eval.evaluate(recommenderBuilder, modelBuilder, model, null, 
//				50,
//				0.8,
//				0.1
//				);
		
//			
		
//		DataModel model = new FileDataModel(new File("inter-matrix"));
//
//	    RecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
//	    // Build the same recommender for testing that we did last time:
//	    RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
//	      @Override
//	      public Recommender buildRecommender(DataModel model) throws TasteException {
//	    	  return new SVDRecommender(model, new ALSWRFactorizer(model, 20, 1.0, 20));	    	  
//	      }
//	    };
//	    // Use 70% of the data to train; test using the other 30%.
//	    double score = evaluator.evaluate(recommenderBuilder, null, model, 0.8, 0.1);
//	    System.out.println(score);
	}
}
