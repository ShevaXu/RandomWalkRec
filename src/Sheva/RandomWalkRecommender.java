package Sheva;

import java.io.*;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.model.BooleanPreference;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A {@link org.apache.mahout.cf.taste.recommender.Recommender} that uses matrix
 * factorization (a projection of users and items onto a feature space)
 */
public final class RandomWalkRecommender extends AbstractRecommender {

	private static final Logger log = LoggerFactory
			.getLogger(RandomWalkRecommender.class);

	private FastByIDMap<FastByIDMap<FastWeight>> allNodes;
	private int numItems;
	private int numUsers;
	private int numArtists;	
	private int[] arriveCount;
	private double[] sumOfWeights;
	
	private final Random random;
	private static final int maxIDs = 100000;
//	private static final long IDofSum = 100000;
	private static final int userOffset = 90000;
	private static final double artistLink = 0.5;
	private static final double itemUserLink = 0.5;
	
	private class FastWeight {
		double weight;
		public FastWeight(double weight) {
			this.weight = weight;
		}
		public double getWeight() {
			return weight;
		}
	}
	
	public RandomWalkRecommender(DataModel dataModel, String graphFile)
			throws TasteException {
		super(dataModel, getDefaultCandidateItemsStrategy());
		numArtists = 0;
		loadGraph(dataModel, graphFile);
		random = RandomUtils.getRandom();
	}

	private Boolean loadGraph(DataModel model, String sourceFile) throws TasteException {
		allNodes = new FastByIDMap<FastByIDMap<FastWeight>>();
		if (sourceFile == null) {
			log.info("No extra graph info!");
			constructUIG(model, false);
//			return false;
		}
		else {
			try {
				BufferedReader br = new BufferedReader(
						new InputStreamReader(new FileInputStream(sourceFile)));			
				while(true) {
					String line = br.readLine();
					if (line == null) break;
					// format: artistID \t songID,songID... 
					String fields[] = line.split("\t");
					if (fields.length < 2) continue;				
					int artistID = Integer.parseInt(fields[0]);
					String songs[] = fields[1].split(",");
					int nSongs = songs.length;
					if (nSongs < 1) continue;
					numArtists++;
					// put in nodes
					double weight = artistLink / (double)nSongs;
//					FastByIDMap<FastWeight> artistMap = new FastByIDMap<FastWeight>();
					allNodes.put((long)artistID, new FastByIDMap<FastWeight>());
					for (int i = 0; i < nSongs; i++) {
						int songID = Integer.parseInt(songs[i]);
						// put in song node
//						FastByIDMap<FastWeight> map = new FastByIDMap<FastWeight>();
//						map.put(artistID, new FastWeight(artistLink));
//						allNodes.put((long)songID, map);					
						allNodes.put((long)songID, new FastByIDMap<FastWeight>(1));
						allNodes.get((long)songID).put((long)artistID, new FastWeight(artistLink));
						//
//						artistMap.put(songID, new FastWeight(weight));
						allNodes.get((long)artistID).put((long)songID, new FastWeight(artistLink));
					}
//					allNodes.put((long)artistID, artistMap);
				}
				log.info("Artist-song graph constructed!");		
				br.close();
			} catch (IOException ioe) {
				System.out.println(ioe.getMessage());
				return false;
			}
			constructUIG(model, true);
		}		
		int totalSize = allNodes.size();
		System.out.println("Artists: " + numArtists + " Users: " + numUsers + 
				" Items from prefs: " + numItems	+ " total: " + totalSize);	
		if (totalSize - numArtists - numUsers > numItems) {
			numItems = totalSize - numArtists - numUsers;
			System.out.println("Final items: " + numItems);
		}
		postProcessGraph();
		log.info("Graph completed");
//		arriveCount = new int[numItems + 1];
		arriveCount = new int[maxIDs];
		return true;
	}
	
	private void constructUIG(DataModel model, boolean extraGraph) throws TasteException {
		log.info("Constructing user-item graph...");
		numItems = model.getNumItems();
		numUsers = model.getNumUsers();
		LongPrimitiveIterator it = model.getUserIDs();
		while(it.hasNext()) {
			long userID = it.nextLong();
			PreferenceArray prefs = model.getPreferencesFromUser(userID);
			int size = prefs.length();
			double weight = 1.0 / (double)size;
			for (int i = 0; i < size; i++) {
				processUIPair(userID + userOffset, prefs.getItemID(i), weight);				
			}
		}
	}
		
	private void processUIPair(long userID, long itemID, double weight) {
		// put in user node
		FastByIDMap<FastWeight> userMap = allNodes.get(userID);
		if (userMap == null) {
//			userMap = new FastByIDMap<FastWeight>();
//			userMap.put(itemID, new FastWeight(weight));
//			allNodes.put(userID, userMap);
			allNodes.put(userID, new FastByIDMap<FastWeight>());
			allNodes.get(userID).put(itemID, new FastWeight(weight));
		}
		else {
			userMap.put(itemID, new FastWeight(weight));
		}
		FastByIDMap<FastWeight> itemMap = allNodes.get(itemID);
		if (itemMap == null) {
//			itemMap = new FastByIDMap<FastWeight>();
//			itemMap.put(userID, new FastWeight(weight / itemUserLink));
//			allNodes.put(itemID, userMap);
			allNodes.put(itemID, new FastByIDMap<FastWeight>());
			allNodes.get(itemID).put(userID, new FastWeight(weight / itemUserLink));
		}
		else {
			itemMap.put(userID, new FastWeight(weight / itemUserLink));
		}
	}
	
	private void postProcessGraph() {
		log.info("PostProcessing graph...");
		sumOfWeights = new double[maxIDs + 1]; 
		// sum up the node weights
		LongPrimitiveIterator itall = allNodes.keySetIterator();
		long nodeID;
		while(itall.hasNext()) {
			nodeID = itall.next();
			FastByIDMap<FastWeight> map = allNodes.get(nodeID);
			double total = 0.0;
			long itemID;
			LongPrimitiveIterator itk = map.keySetIterator();
			while(itk.hasNext()) {
				itemID = itk.next();
				total += map.get(itemID).getWeight();
			}
//			map.put(IDofSum, new FastWeight(total));
			sumOfWeights[(int)nodeID] = total;
		}
	}

	private void randomWalk(long userID, int steps) {
//		int step;
		System.out.println("User " + userID + " starts walk");
		for(int i = 0; i <= numItems; i++)
			arriveCount[i] = 0;
		for(int j = 0; j < steps / 10; j++) {
			startWalk(userID + userOffset, 5);
			//System.out.println("Walk " + (j+1) + "/10 steps");
		}		
	}
	
	private long startWalk(long userID, int maxSteps) {
		int steps = 0;
		long currentID = userID;
		while(currentID != 0) {
			//if (currentID <= numItems)
				arriveCount[(int)currentID]++;
			//System.out.print(currentID + ",");
			currentID =  nextStep(currentID);
			steps++;
			if (steps > maxSteps)
				break;
		}
		return currentID;
	}
	
	private long nextStep(long id) {
		FastByIDMap<FastWeight> map = allNodes.get(id);
//		double totalWeight = map.get(IDofSum).getWeight();
//		totalWeight;
		double thresh = random.nextDouble() * sumOfWeights[(int)id];
		double current = 0.0;
		long itemID = 0;
		LongPrimitiveIterator itk = map.keySetIterator();
		while(itk.hasNext()) {
			itemID = itk.next();
			current += map.get(itemID).getWeight();
			if (current > thresh)
				return itemID;
		}
		return itemID;
	}
	
	@Override
	public List<RecommendedItem> recommend(long userID, int howMany,
			IDRescorer rescorer) throws TasteException {
		Preconditions.checkArgument(howMany >= 1, "howMany must be at least 1");
		log.debug("Recommending items for user ID '{}'", userID);

		// random walk for this user
		long start = System.currentTimeMillis();
		randomWalk(userID, numItems);
		long end = System.currentTimeMillis();
		System.out.println("Walk run time: " + (end - start) + "ms");			
		//******
		
		PreferenceArray preferencesFromUser = getDataModel()
				.getPreferencesFromUser(userID);
		FastIDSet possibleItemIDs = getAllOtherItems(userID,
				preferencesFromUser);

		List<RecommendedItem> topItems = TopItems.getTopItems(howMany,
				possibleItemIDs.iterator(), rescorer, new Estimator(userID));
		log.debug("Recommendations are: {}", topItems);

		return topItems;
	}

	/**
	 * a preference is estimated by computing the dot-product of the user and
	 * item feature vectors
	 */
	@Override
	public float estimatePreference(long userID, long itemID)
			throws TasteException {

		double estimate = (double)arriveCount[(int)itemID] / (double)numItems;

		return (float) estimate;
	}

	private final class Estimator implements TopItems.Estimator<Long> {

		private final long theUserID;

		private Estimator(long theUserID) {
			this.theUserID = theUserID;
		}

		@Override
		public double estimate(Long itemID) throws TasteException {
			return estimatePreference(theUserID, itemID);
		}
	}

	/**
	 * Refresh the data model and factorization.
	 */
	@Override
	public void refresh(Collection<Refreshable> alreadyRefreshed) {

	}

}