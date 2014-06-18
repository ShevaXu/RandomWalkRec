package Sheva;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.BooleanPreference;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public final class OccfEvaluator implements RecommenderIRStatsEvaluator {

	private static final Logger log = LoggerFactory
			.getLogger(OccfEvaluator.class);

	private static final double LOG2 = Math.log(2.0);
	//public static final double CHOOSE_THRESHOLD = Double.NaN;
	private final Random random;

	public OccfEvaluator() {
		random = RandomUtils.getRandom();
	}

	@Override
	public IRStatistics evaluate(RecommenderBuilder recommenderBuilder,
			DataModelBuilder dataModelBuilder, DataModel dataModel,
			IDRescorer rescorer, 
			int at, 
			double trainingPercentage,
			double evaluationPercentage) throws TasteException {

		// check arguments
		Preconditions.checkArgument(recommenderBuilder != null,
				"recommenderBuilder is null");
		Preconditions.checkArgument(dataModel != null, "dataModel is null");
		Preconditions.checkArgument(at >= 1, "at must be at least 1");
		Preconditions.checkArgument(trainingPercentage > 0.0
				&& trainingPercentage < 1.0,
				"Invalid evaluationPercentage: %s", trainingPercentage);
		Preconditions.checkArgument(evaluationPercentage > 0.0
				&& evaluationPercentage <= 1.0,
				"Invalid evaluationPercentage: %s", evaluationPercentage);

		// initialize
		int numItems = dataModel.getNumItems();
		int numUsers = dataModel.getNumUsers();
	    FastByIDMap<PreferenceArray> trainUserPrefs = new FastByIDMap<PreferenceArray>(numUsers);
	    FastByIDMap<PreferenceArray> testUserPrefs = new FastByIDMap<PreferenceArray>(numUsers);
		LongPrimitiveIterator it = dataModel.getUserIDs();
		long start, end;
		FullRunningAverageAndStdDev nDCG = new FullRunningAverageAndStdDev();
		FullRunningAverageAndStdDev mAP = new FullRunningAverageAndStdDev();
		
		// extract training & test set
		start = System.currentTimeMillis();
		while (it.hasNext()) {
			long userID = it.nextLong();
			processOneUser(trainingPercentage, trainUserPrefs, testUserPrefs, userID, dataModel);
		}
		end = System.currentTimeMillis();
		log.info("Extract training & test set in {}ms", end - start);

		DataModel trainingModel = dataModelBuilder == null ? new GenericDataModel(trainUserPrefs)
				: dataModelBuilder.buildDataModel(trainUserPrefs);
		Recommender recommender = recommenderBuilder.buildRecommender(trainingModel);

		// reinitialize it
		it = dataModel.getUserIDs();
		// evaluate MAP
		System.out.println("Evaluation begin!");
		start = System.currentTimeMillis();
		while (it.hasNext()) {
			long userID = it.nextLong();
			if (random.nextDouble() >= evaluationPercentage) {
		        // Skipped
		        continue;
		    }
			try {
				trainingModel.getPreferencesFromUser(userID);
			} catch (NoSuchUserException nsee) {
				continue; // Oops we excluded all prefs for the user -- just
							// move on
			}
			PreferenceArray prefs = testUserPrefs.get(userID);
			if (prefs == null)
				continue;
			int size = prefs.length();
			FastIDSet relevantItemIDs = new FastIDSet(size);
			for (int i = 0; i < size; i++) {
				relevantItemIDs.add(prefs.getItemID(i));
			}
			List<RecommendedItem> recommendedItems = recommender.recommend(
					userID, at, rescorer);
			// calculate AP
//			double ap = 0.0;
//			int hit = 0;
//			for (int i = 0; i < recommendedItems.size(); i++) {
//				RecommendedItem item = recommendedItems.get(i);
//				if (relevantItemIDs.contains(item.getItemID())) {
//					hit++;
//					ap += ((double) hit / (double) (i + 1)); // precision at i
//				}
//			}
//			ap /= (double) size;
//			mAP.addDatum(ap);
//			System.out.println("ap: " + ap);
			// nDCG
			// In computing, assume relevant IDs have relevance 1 and others 0
			double cumulativeGain = 0.0;
			double idealizedGain = 0.0;
			for (int i = 0; i < recommendedItems.size(); i++) {
				RecommendedItem item = recommendedItems.get(i);
				double discount = i == 0 ? 1.0 : 1.0 / log2(i + 1);
				if (relevantItemIDs.contains(item.getItemID())) {
					cumulativeGain += discount;
				}
				if (i < relevantItemIDs.size()) {
					idealizedGain += discount;
				}
			}
			nDCG.addDatum(cumulativeGain / idealizedGain);
			//System.out.println("ndcg: " + cumulativeGain / idealizedGain);
		}
		end = System.currentTimeMillis();

		log.info("Evaluated in {}ms", end - start);		
		log.info("nDCG: {} / {} ", nDCG.getAverage(), nDCG.getStandardDeviation());
		//log.info("nDCG: {} / {} ", mAP.getAverage(), mAP.getStandardDeviation());
		return new MyIRStatic(nDCG.getAverage(), nDCG.getStandardDeviation());
	}
	
	private void processOneUser(double trainingPercentage,
			FastByIDMap<PreferenceArray> trainUserPrefs,
			FastByIDMap<PreferenceArray> testUserPrefs, 
			long userID,
			DataModel dataModel) throws TasteException {
		List<BooleanPreference> trainingPrefs = null;
		List<BooleanPreference> testPrefs = null;
		PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
		int size = prefs.length();
		for (int i = 0; i < size; i++) {
			BooleanPreference newPref = new BooleanPreference(userID, prefs.getItemID(i));
			if (random.nextDouble() < trainingPercentage) {
				if (trainingPrefs == null) {
					trainingPrefs = new ArrayList<BooleanPreference>(2);
				}
				trainingPrefs.add(newPref);
			} else {
				if (testPrefs == null) {
					testPrefs = new ArrayList<BooleanPreference>(2);
				}
				testPrefs.add(newPref);
			}
		}
		if (trainingPrefs != null) {
			trainUserPrefs.put(userID, new GenericUserPreferenceArray(
					trainingPrefs));
			if (testPrefs != null) {
				testUserPrefs.put(userID, new GenericUserPreferenceArray(
						testPrefs));
			}
		}
	}

	private static double log2(double value) {
		return Math.log(value) / LOG2;
	}

}