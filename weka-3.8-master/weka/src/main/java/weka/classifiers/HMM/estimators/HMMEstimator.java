package weka.estimators;

import java.util.Random;

import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.matrix.DoubleVector;

public interface HMMEstimator extends RevisionHandler {

	public int getNumStates();

	public void setNumStates(int NumStates);

	public int getNumOutputs();

	public void setNumOutputs(int NumOutputs) throws Exception;

	public int getOutputDimension();
	/**
	   * parameter prev_state mengatur status HMM sebelumnya
	   * parameter state menyatakan status HMM saat ini
	   * parameter output menghasilkan output HMM
	   * parameter weight (bobot yang telah ditetapkan ke data
	   */
	  void addValue(double prev_state, double state, DoubleVector output, double weight);

	  void addValue(double prev_state, double state, double output, double weight) throws Exception;

	  void addValue0(double state, DoubleVector output, double weight);

	  void addValue0(double state, double output, double weight) throws Exception;


	  /**
	   * mendapatkan probabilitas
	   */
	  double getProbability(double prev_state, double state, DoubleVector output) throws Exception;

	  double getProbability(double prev_state, double state, double output) throws Exception;


	  /**
	   * mendapatkan probabilitas saat step pertama
	   */
	  double getProbability0(double state, DoubleVector output) throws Exception;

	  double getProbability0(double state, double output) throws Exception;
	  
	  int Sample0(Instances sequence, Random generator);
	  
	  int Sample(Instances sequence, int prevState, Random generator);
	  
	  void calculateParameters() throws Exception;
}
