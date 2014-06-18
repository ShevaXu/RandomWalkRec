package Sheva;

import java.io.Serializable;

import org.apache.mahout.cf.taste.eval.IRStatistics;

import com.google.common.base.Preconditions;

public final class MyIRStatic implements IRStatistics, Serializable {

	private final double stddev;
	private final double ndcg;

	// MyIRStatic(double precision, double recall, double fallOut, double ndcg)
	// {
	MyIRStatic(double ndcg, double stddev) {
		Preconditions.checkArgument(ndcg >= 0.0 && ndcg <= 1.0,
				"Illegal nDCG: " + ndcg);
		this.stddev = stddev;
		this.ndcg = ndcg;
	}

	@Override
	public double getNormalizedDiscountedCumulativeGain() {
		return ndcg;
	}

	@Override
	public double getF1Measure() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getFNMeasure(double arg0) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getFallOut() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getPrecision() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getRecall() {
		// TODO Auto-generated method stub
		return 0;
	}
	
	public double getStdDev() {
		return stddev;
	}
}