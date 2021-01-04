package weka.classifiers.bayes;


import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;
import java.lang.Math;
import java.util.Random;
import weka.classifiers.*;
import weka.classifiers.RandomizableClassifier;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.matrix.DoubleVector;
import weka.core.matrix.Matrix;
import weka.estimators.DiscreteHMMEstimator;
import weka.estimators.HMMEstimator;
import weka.estimators.MultivariateNormalEstimator;
import weka.estimators.MultivariateNormalHMMEstimator;

public class HMM extends weka.classifiers.RandomizableClassifier implements weka.core.OptionHandler, weka.core.MultiInstanceCapabilitiesHandler {

	
	
	protected class ProbabilityTooSmallException extends Exception
	{
		/**
		 * tampilan antarmuka Serializable ke kelas DoubleVector Weka
		 */
		private static final long serialVersionUID = -2706223192260478060L;

		ProbabilityTooSmallException(String s)
		{
			super(s);
		}
	}
	
	/** inisiasi jumlah kelas yang ada */
	protected int m_NumStates=6;
	protected int m_NumOutputs;
	protected int m_OutputDimension = 1;
	protected boolean m_Numeric = false;
	protected double m_IterationCutoff = 0.01;
	protected int m_SeqAttr = -1;
	
	/*
	 * mendapatkan index antribut data yang isinya adalah sequence data
	 */
	public int getSequenceAttribute()
	{
		return m_SeqAttr;
	}
	
	protected Random m_rand = null;
	
	protected double minScale = 1.0E-200;

	protected boolean m_RandomStateInitializers = false;
	
	/**
	 * mengetahui apakah probabilitas HMM diinisialisasi secara acak, jika tidak secara acak
	 * maka HMM ditetapkan menggunakan clustering pada dataset
	 */
	public boolean isRandomStateInitializers() {
		return m_RandomStateInitializers;
	}

	/**
	 * mengatur apakah probabilitas HMM diinisialisasi secara acak, jika salah maka
	 * ditetapkan menggunakan clustering pada dataset
	 *
	 * maksud dari randomStateInitializers adalah jika ia bernilai benar, maka probabilitas HMM akan
	 * diinisialisasi secara acak, jika salah maka akan diinisialisasi menggunakan clustering
	 */
	public void setRandomStateInitializers(boolean randomStateInitializers) {
		this.m_RandomStateInitializers = randomStateInitializers;
	}


	protected boolean m_Tied = false;

	/**
	 * mendapatkan apakah matriks kovariansi dari HMM guassian terikat.
	 * Kovariansi terikat sama untuk semua status HMM.
	 * Menggunakan kovarian terikat memungkinkan untuk mempelajari model lebih mudah
	 * akan tetapi data membatasi kelas model yang dapat dipelajarinya.
	 */
	public boolean isTied() {
		return m_Tied;
	}

	/**
	 * menentukan apakah matriks kovariansi HMM guassian terikat.
	 * maksud dari m_tied = tied adalah inisiasi tied jika ia bernilai true kovarian akan terikat
	 */
	public void setTied(boolean tied) {
		m_Tied = tied;
	}
	
	
	/**
	 * jenis matriks kovarians untuk HMM gaussian :
	 * 		- Matriks LENGKAP memungkinkan ketergantungan arbitrer antar variabel,
	 * 		- Sebuah matriks DIAGONAL mengasumsikan bahwa semua variabel tidak bergantung.
	 * 		- Model SPHERICAL mengasumsikan bahwa variabel independen dan semua memiliki varian yang sama. Kovarian SPHERICAL menyiratkan lebih terbata
	 * 		set model daripada kovarian DIAGONAL yang lebih terbatas dari matriks FULL. Model yang dibatasi membutuhkan lebih sedikit data untuk dipelajari
	 * 		tetapi mungkin tidak dapat membuat model semua fitur data.
	 */
	public static final Tag [] TAGS_COVARIANCE_TYPE = {
	    new Tag(MultivariateNormalEstimator.COVARIANCE_FULL, "Full matrix (tidak ada batasan)"),
	    new Tag(MultivariateNormalEstimator.COVARIANCE_DIAGONAL, "Diagonal matrix (tidak ada korelasi antara atribut data)"),
	    new Tag(MultivariateNormalEstimator.COVARIANCE_SPHERICAL, "Spherical matrix (semua atribut memiliki varian yang sama)"),
	};
	
	protected int m_CovarianceType = MultivariateNormalEstimator.COVARIANCE_FULL;

	public SelectedTag getCovarianceType() {
		return new SelectedTag(m_CovarianceType, TAGS_COVARIANCE_TYPE);
	}

	public void setCovarianceType(SelectedTag covarianceType) {
		if (covarianceType.getTags() == TAGS_COVARIANCE_TYPE) {
			m_CovarianceType = covarianceType.getSelectedTag().getID();
		}
	}

	protected boolean m_LeftRight = false;
	
	/**
	 * mengetahui apakah HMM memiliki struktur status left-right
	 * Dalam HMM left-right, urutan status selalu melewati urutan tetap (mungkin beberapa bagian state telah hilang).
	 * Alternatifnya adalah  menggunakan model default yang mana transisi secara sewenang-wenang dapat dilakukan.
	 */
	public boolean isLeftRight() {
		return m_LeftRight;
	}

	/**
	 * parameter leftRight bernilai jika true makan hmm menggunakan struktur left-right,
	 * jika salah maka menggunakan model default
	 */
	public void setLeftRight(boolean leftRight) {
		this.m_LeftRight = leftRight;
	}

	/*
	 * mendapatkan dimensi keluarkan dari HMM
	 */
	public int getOutputDimension() {
		return m_OutputDimension;
	}

	protected void setOutputDimension(int OutputDimension) {
		m_OutputDimension = OutputDimension;
	}

	/*
	 * mendapatkan apakah keluarannya numerik
	 * return true jika numerik dan false jika nominal
	 */
	public boolean isNumeric() {
		return m_Numeric;
	}

	protected void setNumeric(boolean Numeric) {
		m_Numeric = Numeric;
		if (m_Numeric)
			setIterationCutoff(0.0001);
		else
			setIterationCutoff(0.01);
	}

	/*
	 * mendapatkan nilai cut off untuk menghentikan iterasi EM.
	 * Iterasi EM akan berhenti ketika kemungkinan ada perubahan relatif
	 * antara dua iterasi berikutnya berada di bawah batas nilai cut off
	 * Kelas EM (ekspektasi maksimalisasi) sederhana.
	 * EM memberikan distribusi probabilitas untuk setiap instance yang menunjukkan probabilitasnya milik masing-masing cluster.
	 * EM dapat memutuskan berapa banyak cluster yang akan dibuat dengan validasi silang, atau Anda dapat menentukan
	 * apriori berapa banyak cluster yang akan dibuat.
	 */
	public double getIterationCutoff() {
		return m_IterationCutoff;
	}

	/*
	 * menetapkan nilai cut off untuk menghentikan iterasi EM.
	 * Iterasi EM akan berhenti ketika kemungkinan ada perubahan relatif
	 * antara dua iterasi berikutnya berada di bawah batas nilai cut off
	 */
	public void setIterationCutoff(double iterationCutoff) {
		m_IterationCutoff = iterationCutoff;
	}

	protected HMMEstimator estimators[];

	/**
	 * mendapatkan jumlah kelas untuk pengklasifikasi HMM.
	 * Nilai diatur secara otomatis dari data
	 */
	public int getNumClasses()
	{
		if(estimators == null)
			return 0; 
		else
			return estimators.length;
	}
	
	/**
	 * mendapatkan jumlah nilai status tersembunyi di HMM
	 */
	public int getNumStates() {
		return m_NumStates;
	}

	/**
	 * mengatur jumlah status tersembunyi di HMM.
	 * Jumlah status adalah parameter utama HMM dan harus disetel (tidak dipelajari dari data).
	 * parameter numStates adalah jumlah state yang akan digunakan
	 */
	public void setNumStates(int numStates) {
		m_NumStates = numStates;
	}
	
	/**
	 * mendapatkan jumlah nilai keluaran (observasi) yang berbeda untuk data nominal
	 * hal Ini diatur secara otomatis dari data.
	 */
	public int getNumOutputs()
	{
		return m_NumOutputs;
	}
	
	protected void setNumOutputs(int numOutputs)
	{
		m_NumOutputs = numOutputs;
	}

	/**
	 * parameter classId adalah kelas probabilitas yang telah disiapkan
	 * parameter state adalah state saat ini
	 * parameter output adalah keluaran
	 * parameter prob adalah nilai pobabilitas
	 */
	public void setProbability0(int classId, double state, DoubleVector output,
			double prob) {
		estimators[classId].addValue0(state, output, prob);
	}

	/**
	 * parameter classId adalah kelas probabilitas yang telah disiapkan
	 * parameter state adalah state saat ini
	 * parameter output adalah keluaran
	 * parameter prob adalah nilai pobabilitas
	 */
	public void setProbability(int classId, double prevState, double state, DoubleVector output,
			double prob) {
		estimators[classId].addValue(prevState, state, output, prob);
	}
	
	protected double likelihoodFromScales(double scales[])
	{
		double lik = 0.0f;
		for (int i = 0; i < scales.length; i++)
			if(Math.abs((scales[i])) > 1.0E-32)
				lik += Math.log(scales[i]);
			else
				lik += Math.log(1.0E-32);
		return lik;
	}
	
	protected double [] forward(HMMEstimator hmm, Instances sequence, double alpha[][]) throws Exception
	{
		double scales [] =  new double [sequence.numInstances()];
		

		scales[0] = 0.0f;
		DoubleVector output = new DoubleVector(sequence.instance(0).numAttributes());
		for(int i = 0; i < sequence.instance(0).numAttributes(); i++)
			output.set(i, sequence.instance(0).value(i));
		for (int s = 0; s < m_NumStates; s++)
		{
			
			alpha[0][s] = hmm.getProbability0(s, output);
			scales[0] += alpha[0][s];
		}
		
		// melakukan scaling
		if(Math.abs(scales[0]) > minScale)
		{
			for (int s = 0; s < m_NumStates; s++)
				alpha[0][s] /= scales[0];
		}
		else
		{
			throw new ProbabilityTooSmallException("time step 0 probability " + scales[0]);
		}
		

		for (int t = 1; t < sequence.numInstances(); t++)
		{
			output = new DoubleVector(sequence.instance(t).numAttributes());
			for(int i = 0; i < sequence.instance(t).numAttributes(); i++)
				output.set(i, sequence.instance(t).value(i));
			//System.out.println(output);
			scales[t] = 0.0f;
			for(int s = 0; s < m_NumStates; s++)
			{
				alpha[t][s] = 0.0f;
				for (int ps = 0; ps < m_NumStates; ps++)
				{
					//int output = (int) sequence.instance(t).value(0);
					
					alpha[t][s] += alpha[t-1][ps]*hmm.getProbability(ps, s, output);
				}
				scales[t] += alpha[t][s];
			}
			// melakukan scaling
			if(Math.abs(scales[t]) > minScale)
			{
				for (int s = 0; s < m_NumStates; s++)
					alpha[t][s] /= scales[t];
			}
			else
			{
				throw new ProbabilityTooSmallException("time step " + t + " probability " + scales[t]);
			}
		}
		
		return scales;
	}
	
	/**
	 * menjalankan algoritma maju (perhitungan likelihood) pada urutan tertentu
	 * parameter hmm digunakan untuk mengevaluasi
	 * parameter sequence digunakan untuk mengurutkan nilai yang akan dievaluasi
	 */
	protected double forward(HMMEstimator hmm, Instances sequence) throws Exception
	{
		double alpha[][] = new double[sequence.numInstances()][m_NumStates];
		double scales [] =  forward(hmm, sequence, alpha);
		return likelihoodFromScales(scales);
	}
	
	protected double [] forwardBackward(HMMEstimator hmm, Instances sequence, double alpha[][], double beta[][]) throws Exception
	{
		double scales [] =  forward(hmm, sequence, alpha);
		

		for (int s = 0; s < getNumStates(); s++)
		{
			beta[sequence.numInstances()-1][s] = 1.0f;
			if (Double.isInfinite(beta[sequence.numInstances()-1][s]) || Double.isNaN(beta[sequence.numInstances()-1][s]))
				throw new Exception("hasil beta untuk final step adalah NaN");
		}
		
		
		// langkah mundur melalui sisa sequence
		for (int t = sequence.numInstances()-2; t >= 0; t--)
		{
			for (int s = 0; s < getNumStates(); s++)
			{
				beta[t][s] = 0.0f;
				for (int ns = 0; ns < getNumStates(); ns++)
				{
					DoubleVector output = new DoubleVector(sequence.instance(t+1).numAttributes());
					for(int i = 0; i < sequence.instance(t+1).numAttributes(); i++)
						output.set(i, sequence.instance(t+1).value(i));
					double p = hmm.getProbability(s, ns, output);
					beta[t][s] += beta[t+1][ns]*p;
					if (Double.isInfinite(beta[t][s]) || Double.isNaN(beta[t][s]))
						throw new Exception("nilai beta tanpa skala adalah NaN");
				}
				
					
			}
			if(Math.abs(scales[t+1]) > minScale)
			{
				for (int s = 0; s < getNumStates(); s++)
				{
					beta[t][s] /= scales[t+1];
					if (Double.isInfinite(beta[t][s]) || Double.isNaN(beta[t][s]))
						throw new Exception("nilai beta dengan skala adalah NaN");
				}
			}
			else
			{
				throw new ProbabilityTooSmallException("time step " + (t+1) + " probabilit " + scales[t+1]);
			}
		}
		
		return scales;
	}
	
	protected double forwardBackward(HMMEstimator hmm, Instances sequence) throws Exception
	{
		double alpha[][] = new double[sequence.numInstances()][m_NumStates];
		double beta[][] = new double[sequence.numInstances()][m_NumStates];
		double scales [] = forwardBackward(hmm, sequence, alpha, beta);
		return likelihoodFromScales(scales);
	}
	
	/**
	 * mengevaluasi probabilitas state beberapa bagian untuk urutan tertentu
	 *
	 * parameter classId adalah kelas untuk melakukan evaluasi
	 * parameter instace adalah  contoh data (sequence) yang akan dievaluasi
	 * mengembalikan  array probabilitas dua dimensi. Item [i] [j] memberikan kemungkinan berada di state j untuk item ke-i didalam sequence.
	 */
	public double [][] probabilitiesForInstance(int classId, weka.core.Instance instance) throws Exception 
	{
		Instances sequence = instance.relationalValue(m_SeqAttr);
		double alpha[][] = new double[sequence.numInstances()][m_NumStates];
		double beta[][] = new double[sequence.numInstances()][m_NumStates];
		double gamma[][] = new double[sequence.numInstances()][m_NumStates];
		double scales [] = forwardBackward(estimators[classId], sequence, alpha, beta);
		double scale = Math.exp(likelihoodFromScales(scales));
		for (int i = 0; i < gamma.length; i++)
		{
			//System.out.print("gamma {");
			for (int j = 0; j < gamma[i].length; j++)
			{
				gamma[i][j] = alpha[i][j]*beta[i][j];///scale;
				//System.out.print(gamma[i][j] + " ");
			}

		}
		return gamma;
	}
	
	/**
	 * get the probabilities from a particular sequence
	 * 
	 * @param instance the data instance (sequence)
	 * @return an array of probabilities (doubles). The nth value is the probability that the sequence is of the nth class.
	 *
	 * mendapatkan probabilitas dari sequence tertentu
	 *
	 * parameter instance berguna untuk instance data (sequence)
	 * mengembalikan nilai array probabilitas. Nilai ke-n adalah probabilitas bahwa urutan tersebut dari kelas ke-n.
	 */
	public double[] distributionForInstance(weka.core.Instance instance) throws Exception {
		
		if(estimators == null)
		{
			double result[] = {0.5,0.5};
			return result;
		}
		
		 double [] result = new double[estimators.length];
		 double sum = 0.0;

		 if (m_SeqAttr < 0)
		 {
			 for (int j = 0; j < estimators.length; j++)
			 {
				 result[j] = 1.0;
				 sum += result[j];
			 };
		 }
		 else
		 {
			 Instances seq = instance.relationalValue(m_SeqAttr);
			 for (int j = 0; j < estimators.length; j++)
			 {
				try
				{
					result[j] = Math.exp(forward(estimators[j], seq));
				}
				catch(ProbabilityTooSmallException e)
				{
					result[j] = 0;
				}
				 sum += result[j];
			 };
		 }
		 
		 if(Math.abs(sum) > 0.0000001)
		 {
			 for (int i = 0; i < estimators.length; i++)
				 result[i] /= sum;
		 }
		return result;
	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return super.clone();
	}

	/**
	 * Mengembalikan string yang menjelaskan classifier
	 * mengembailan deskripsi pengklasifikasi yang cocok untuk Menampilkan di gui
	 */
	  public String globalInfo() {
	    return "Kelas untuk pengklasifikasi Hidden Markov Model..";
	  }
	
	@Override
	public String[] getOptions() {
		String [] options = new String [10];
	    int current = 0;

	    options[current++] = "-S " + getNumStates();
	    
	    options[current++] = "-I " + getIterationCutoff();
	    
	    switch (m_CovarianceType)
	    {
	    case MultivariateNormalEstimator.COVARIANCE_FULL:
	    	options[current++] = "-C FULL";
	    	break;
	    case MultivariateNormalEstimator.COVARIANCE_DIAGONAL:
	    	options[current++] = "-C DIAGONAL";
	    	break;
	    case MultivariateNormalEstimator.COVARIANCE_SPHERICAL:
	    	options[current++] = "-C SPHERICAL";
	    	break;
	    }

	    
	    options[current++] = "-D"  + isTied();
	    
	    options[current++] = "-L" + isLeftRight();
	    
	    options[current++] = "-R" + isRandomStateInitializers();
	
	    while (current < options.length) {
	      options[current++] = "";
	    }
	    return options;
	}

	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(4);
		
	    newVector.addElement(
	              new Option("\tStates: jumlah state HMM yang digunakan\n",
	                         "S", 1,"-S"));

	    newVector.addElement(
	              new Option("\tIteration Cutoff: the proportional minimum change of likelihood\n"
	                         +"\tdimana digunakan untuk menghentikan iterasi EM ",
	                         "I", 1,"-I"));

	    newVector.addElement(
	              new Option("\tCovariance Type: whether the covariances of gaussian\n"
	                         +"\toutputs harus berupa matriks penuh atau dibatasi pada matriks diagonal\n"
	                         +"\tatau matriks spherical",
	                         "C", 1,"-C"));

	    newVector.addElement(
	              new Option("\tTied Covariance: whether the covariances of gaussian\n"
	                         +"\toutputs are tied to be the same across all outputs ",
	                         "D", 1,"-D"));

	    newVector.addElement(
	              new Option("\tLeft Right: whether the state transitions are constrained\n"
	                         +"\tto go only to the next state in numerical order ",
	                         "L", 1,"-L"));

	    newVector.addElement(
	              new Option("\tRandom Initialisation: whether the state transition probabilities are intialized randomly\n"
	                         +"\t(jika false, data akan diinisialisasi dengan menggunakan k-means clustering) ",
	                         "R", 1,"-R"));

	    
	    return newVector.elements();
	}

	
	@Override
	public void setOptions(String[] options) throws Exception {
		
		String cutoffString = Utils.getOption('I', options);
	    if (cutoffString.length() != 0) 
	      setIterationCutoff(Double.parseDouble(cutoffString));
	    
	    String statesString = Utils.getOption('S', options);
	    if (statesString.length() != 0) 
	      setNumStates(Integer.parseInt(statesString));
	    
	    String covTypeString = Utils.getOption('C', options);
	    if (covTypeString.length() != 0) 
	    {
	    	if (covTypeString.equals("FULL"))
	    		setCovarianceType(new SelectedTag(MultivariateNormalEstimator.COVARIANCE_FULL, TAGS_COVARIANCE_TYPE));
	    	if (covTypeString.equals("DIAGONAL"))
	    		setCovarianceType(new SelectedTag(MultivariateNormalEstimator.COVARIANCE_DIAGONAL, TAGS_COVARIANCE_TYPE));
	    	if (covTypeString.equals("SPHERICAL"))
	    		setCovarianceType(new SelectedTag(MultivariateNormalEstimator.COVARIANCE_SPHERICAL, TAGS_COVARIANCE_TYPE));
	    }
	      
	    if(Utils.getFlag('D', options))
	    	setTied(true);

	    if(Utils.getFlag('L', options))
	    	setLeftRight(true);
	    
	    if(Utils.getFlag('R', options))
	    	setRandomStateInitializers(true);

	    Utils.checkForRemainingOptions(options);
	}


	private static final long serialVersionUID = 1959669739718119361L;

	  public Capabilities getCapabilities() {
	    Capabilities result = super.getCapabilities();
	    result.disableAll();

	    // atribut
	    result.enable(Capability.RELATIONAL_ATTRIBUTES);
	    result.enable(Capability.NOMINAL_ATTRIBUTES);
	    result.enable(Capability.NUMERIC_ATTRIBUTES);
	    result.enable(Capability.ONLY_MULTIINSTANCE);

	    // class
	    result.enable(Capability.NOMINAL_CLASS);


	    return result;
	  }

	@Override
	public Capabilities getMultiInstanceCapabilities() {
		Capabilities result = new Capabilities(this);
		result.disableAll();
		
		result.enable(Capability.NOMINAL_ATTRIBUTES);
	    result.enable(Capability.NUMERIC_ATTRIBUTES);
		
		result.enable(Capability.NO_CLASS);
		
		return result;
	}
	  
	protected double EMStep(Instances data) throws Exception
	{
		double lik = 0.0f;
		boolean hasUpdated = false;

		//menyiapkan estimators terbaru yang akan disimpan
		HMMEstimator newEstimators[] = new HMMEstimator[data.numClasses()];;
		for (int i = 0; i < data.numClasses(); i++)
		{
			if(isNumeric())
			{
				MultivariateNormalHMMEstimator est =new MultivariateNormalHMMEstimator(getNumStates(), false);
				est.copyOutputParameters((MultivariateNormalHMMEstimator)estimators[i]);
				newEstimators[i]= est;
			}
			else
			{
				newEstimators[i]=new DiscreteHMMEstimator(getNumStates(), getNumOutputs(), false);
			}
		}
		
		
		
		int numS0 = 0;
		int numS1 = 0;
		for (int i = 0; i < data.numInstances(); i++)
		{
			Instance inst = data.instance(i);
			
			// lewati jika nilai yang relevan telah hilang
			if(inst.isMissing(m_SeqAttr) || inst.classIsMissing())
				continue;
			

			Instances sequence = inst.relationalValue(m_SeqAttr);
			double alpha[][] = new double[sequence.numInstances()][m_NumStates];
			double beta[][] = new double[sequence.numInstances()][m_NumStates];
			
			int classNum = (int) inst.value(data.classIndex());
			
			//System.out.println("****** class " + classNum + " *******");
			HMMEstimator hmm = estimators[classNum];
			
			double scales [];
			try
			{
				scales = forwardBackward(hmm, sequence, alpha, beta);
			}
			catch(ProbabilityTooSmallException e)
			{
				continue;
			}
			double PX = this.likelihoodFromScales(scales);
			lik += PX;
			PX = Math.exp(PX);
			

			double sumGamma = 0.0;
			DoubleVector output = new DoubleVector(sequence.instance(0).numAttributes());
			for(int a = 0; a < sequence.instance(0).numAttributes(); a++)
				output.set(a, sequence.instance(0).value(a));

			double gamma[][] = new double[getNumStates()][getNumStates()];
			for (int s = 0; s < getNumStates(); s++)
			{
				gamma[0][s] = alpha[0][s]*beta[0][s];
				sumGamma += gamma[0][s];
				
			}
			for (int s = 0; s < getNumStates(); s++)
			{
				if(sumGamma > minScale)
					newEstimators[classNum].addValue0(s, output, gamma[0][s]/sumGamma);
				
				if(Double.isInfinite(gamma[0][s]) || Double.isNaN(gamma[0][s]))
					throw new Exception("Output of the forward backward algorithm gives a NaN");
			}

			for (int t = 1; t < sequence.numInstances(); t++)
			{
				sumGamma = 0.0;
				output = new DoubleVector(sequence.instance(t).numAttributes());
				for(int a = 0; a < sequence.instance(t).numAttributes(); a++)
					output.set(a, sequence.instance(t).value(a));
				for (int s = 0; s < getNumStates(); s++)
					for (int ps = 0; ps < getNumStates(); ps++)
					{
						gamma[ps][s] = alpha[t-1][ps]*hmm.getProbability(ps, s, output)*beta[t][s]*scales[t];
						sumGamma += gamma[ps][s];
					}
				for (int s = 0; s < getNumStates(); s++)
					for (int ps = 0; ps < getNumStates(); ps++)
					{
						if(sumGamma > minScale)
						{
							if(classNum == 0 && s == 1 && gamma[ps][s]/sumGamma > 0.01)
							{
									numS1 += 1;
							}
							if(classNum == 0 && s == 0 && gamma[ps][s]/sumGamma > 0.01)
							{
								numS0 += 1;
							}
							newEstimators[classNum].addValue(ps, s, output, gamma[ps][s]/sumGamma);
						}
						
						// check nilai numerik yang tidak diketahui
						if(Double.isInfinite(gamma[ps][s]) || Double.isNaN(gamma[ps][s]))
							throw new Exception("Output of the forward backward algorithm gives a NaN");

					}
			}
			hasUpdated = true;
		}
		// update estimators
		if(hasUpdated)
		{
			estimators = newEstimators;
			for (int i = 0; i < estimators.length; i++)
				estimators[i].calculateParameters();
		}
		else
			throw new Exception("gagal mengupdate EM step");
		return lik/data.numInstances();
	}
	
	/*
	 * Initialize the hmm estimators prior to learning
	 * 
	 * @param numClasses the number of classes (i.e. the number of estimators)
	 * parameter numClass berisi jumlah kelas
	 * parameter data adalah data yang akan digunakan untuk inisiasi
	 */
	public void initEstimators(int numClasses, Instances data) throws Exception
	{
		if(isNumeric())
			initEstimatorsMultivariateNormal(numClasses, null, null, null, null, data);
		else	
			initEstimatorsUnivariateDiscrete(numClasses, null, null, null);
	}
	
	protected double[][] initState0ProbsUniform(int numClasses)
	{
		double [][] state0Probs = new double[numClasses][getNumStates()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
			{
				state0Probs[i][j] = 1;
			}
		return state0Probs;
	}
	
	protected double[][] initState0ProbsRandom(int numClasses, Random rand)
	{
		double [][] state0Probs = new double[numClasses][getNumStates()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
			{
				state0Probs[i][j] = rand.nextInt(100);
			}
		return state0Probs;
	}
	
	protected double[][] initState0ProbsLeftRight(int numClasses)
	{
		double [][] state0Probs = new double[numClasses][getNumStates()];
		for (int i = 0; i < numClasses; i++)
		{
			state0Probs[i][0] = 1;
			for (int j = 1; j < getNumStates(); j++)
			{
				state0Probs[i][j] = 0;
			}
		}
		return state0Probs;
	}
	
	protected double[][][] initStateProbsUniform(int numClasses)
	{
		double [][][] stateProbs = new double[numClasses][getNumStates()][getNumStates()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
				for (int k = 0; k < getNumStates(); k++)
				{
					if(j == k)
						stateProbs[i][j][k] = 10;
					else
						stateProbs[i][j][k] = 1;
				}
		return stateProbs;
	}
	
	protected double[][][] initStateProbsRandom(int numClasses, Random rand)
	{
		double [][][] stateProbs = new double[numClasses][getNumStates()][getNumStates()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
				for (int k = 0; k < getNumStates(); k++)
				{
					stateProbs[i][j][k] = rand.nextInt(100);
				}
		return stateProbs;
	}
	
	protected double[][][] initStateProbsLeftRight(int numClasses)
	{
		double [][][] stateProbs = new double[numClasses][getNumStates()][getNumStates()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
			{
				for (int k = 0; k < getNumStates(); k++)
				{
					stateProbs[i][j][k] = 0;
				}
			}
		for (int i = 0; i < numClasses; i++)
		{
			for (int j = 0; j < getNumStates()-1; j++)
			{
				stateProbs[i][j][j]   = 90;
				stateProbs[i][j][j+1] = 10;
			}
			stateProbs[i][getNumStates()-1][getNumStates()-1]   = 100;	
		}
		return stateProbs;
	}
	
	protected double[][][] initDiscreteOutputProbsRandom(int numClasses, Random rand)
	{
		double [][][] outputProbs = new double[numClasses][getNumStates()][getNumOutputs()];
		for (int i = 0; i < numClasses; i++)
			for (int j = 0; j < getNumStates(); j++)
				for (int k = 0; k < getNumOutputs(); k++)
				{
					outputProbs[i][j][k] = rand.nextInt(100);
				}
		return outputProbs;
	}
	
	protected void initGaussianOutputProbsRandom(int numClasses, DoubleVector outputMeans[][], Matrix outputVars[][])
	{
		if (outputMeans == null)
		{
			outputMeans = new DoubleVector[numClasses][getNumStates()];
			for (int i = 0; i < numClasses; i++)
				for (int j = 0; j < getNumStates(); j++)
					outputMeans[i][j] = DoubleVector.random(getOutputDimension());			
		}
		
		if (outputVars == null)
		{
			outputVars = new Matrix[numClasses][getNumStates()];
			for (int i = 0; i < numClasses; i++)
				for (int j = 0; j < getNumStates(); j++)
				{
					outputVars[i][j] = Matrix.identity(getOutputDimension(),getOutputDimension());	
					outputVars[i][j].timesEquals(10.0);
				}
		}
		
		for (int i = 0; i < numClasses; i++)
		{
			for (int j = 0; j < getNumStates(); j++)
			{
				MultivariateNormalHMMEstimator est = (MultivariateNormalHMMEstimator)(estimators[i]);
				est.setOutputMean(j,outputMeans[i][j]);
				est.setOutputVariance(j,outputVars[i][j]);
			}
		}
	}
	
	protected void initGaussianOutputProbsAllData(int numClasses, Instances data, DoubleVector outputMeans[][], Matrix outputVars[][]) throws Exception
	{
		MultivariateNormalEstimator [] ests = new MultivariateNormalEstimator[numClasses];
		for (int i = 0; i < numClasses; i++)
			ests[i] = new MultivariateNormalEstimator();
		
		m_SeqAttr = -1;
		m_NumOutputs = 0;
		for(int i = 0; i < data.numAttributes(); i++)
		{
			Attribute attr = data.attribute(i);
			if(attr.isRelationValued())
			{
				if(attr.relation().attribute(0).isNominal())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
				}
				if(attr.relation().attribute(0).isNumeric())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
					this.setNumeric(true);
				}
				break;
			}
		}
		
		for (int i = 0; i < data.numInstances(); i++)
		{
			Instance inst = data.instance(i);
			
			// lewati jika nilai yang relevan hilang
			if(inst.isMissing(m_SeqAttr) || inst.classIsMissing())
				continue;
			

			Instances sequence = inst.relationalValue(m_SeqAttr);
			
			int classNum = (int) inst.value(data.classIndex());
			
			for (int j = 0; j < sequence.numInstances(); j++)
			{
				DoubleVector output = new DoubleVector(sequence.instance(j).numAttributes());
				for(int a = 0; a < sequence.instance(j).numAttributes(); a++)
					output.set(a, sequence.instance(j).value(a));
			
				ests[classNum].addValue(output, 1.0);
			}
		}
		for (int i = 0; i < numClasses; i++)
			ests[i].calculateParameters();
		
		for (int i = 0; i < numClasses; i++)
		{
			for (int j = 0; j < getNumStates(); j++)
			{
				MultivariateNormalHMMEstimator est = (MultivariateNormalHMMEstimator)(estimators[i]);
				if(outputMeans == null)
					est.setOutputMean(j,ests[i].getMean());
				else
					est.setOutputMean(j, outputMeans[i][j]);
				if(outputVars == null)
					est.setOutputVariance(j,ests[i].getVariance());
				else
					est.setOutputVariance(j, outputVars[i][j]);
			}
		}
	}
	
	protected void initGaussianOutputProbsCluster(int numClasses, Instances data, DoubleVector outputMeans[][], Matrix outputVars[][]) throws Exception
	{
		
		m_SeqAttr = -1;
		m_NumOutputs = 0;
		for(int i = 0; i < data.numAttributes(); i++)
		{
			Attribute attr = data.attribute(i);
			if(attr.isRelationValued())
			{
				if(attr.relation().attribute(0).isNominal())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
				}
				if(attr.relation().attribute(0).isNumeric())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
				}
				break;
			}
		}
			
		Instances [] flatdata = new Instances[numClasses];
		for (int i = 0; i < data.numInstances(); i++)
		{
			Instance inst = data.instance(i);
			
			// lewati jika nilai yang relevan hilang
			if(inst.isMissing(m_SeqAttr) || inst.classIsMissing())
				continue;

			Instances sequence = inst.relationalValue(m_SeqAttr);
			
			int classNum = (int) inst.value(data.classIndex());
			if(flatdata[classNum] == null)
			{
				flatdata[classNum] = new Instances(sequence, sequence.numInstances());
			}
			
			for (int j = 0; j < sequence.numInstances(); j++)
			{
				flatdata[classNum].add(sequence.instance(j));

			}
			
		}
		
		SimpleKMeans [] kmeans = new SimpleKMeans[numClasses];
		for (int i = 0; i < numClasses; i++)
		{
			kmeans[i] = new SimpleKMeans();
			kmeans[i].setNumClusters(getNumStates());
			kmeans[i].setDisplayStdDevs(true);
			kmeans[i].buildClusterer(flatdata[i]);
		}
		
		
		for (int i = 0; i < numClasses; i++)
		{
			Instances clusterCentroids = kmeans[i].getClusterCentroids();
			Instances clusterStdDevs = kmeans[i].getClusterStandardDevs();
			for (int j = 0; j < getNumStates(); j++)
			{
				MultivariateNormalHMMEstimator est = (MultivariateNormalHMMEstimator)(estimators[i]);
				if(outputMeans == null)
				{
					DoubleVector mean = new DoubleVector(clusterCentroids.instance(j).numAttributes());
					for(int a = 0; a < clusterCentroids.instance(j).numAttributes(); a++)
						mean.set(a, clusterCentroids.instance(j).value(a));
					
					System.out.println("Mean " + j + " " + mean);
					est.setOutputMean(j,mean);
				}
				else
				{
					est.setOutputMean(j, outputMeans[i][j]);
				}
				if(outputVars == null)
				{
					int n = clusterStdDevs.instance(j).numAttributes();
					Matrix sigma = new Matrix(n,n, 0.0);
					for(int a = 0; a < n; a++)
					{
						//double s = clusterCentroids.instance(j).value(a);
						double s = clusterStdDevs.instance(j).value(a);
						sigma.set(a, a, s*s);
					}
					est.setOutputVariance(j,sigma);
				}
				else
				{
					est.setOutputVariance(j, outputVars[i][j]);
				}
			}
		}
	}
	
	public void initEstimatorsUnivariateDiscrete(int numClasses, double state0Probs[][], double stateProbs[][][], double outputProbs[][][]) throws Exception
	{
		estimators = new HMMEstimator[numClasses];
		
		// random initialization
		Random rand = new Random(getSeed());
		if (state0Probs == null)
		{
			if(isLeftRight())
				state0Probs = initState0ProbsLeftRight(numClasses);
			else if(isRandomStateInitializers())
				state0Probs = initState0ProbsRandom(numClasses, rand);
			else
				state0Probs = initState0ProbsUniform(numClasses);
		}

		if (stateProbs == null)
		{
			if(isLeftRight())
				stateProbs = initStateProbsLeftRight(numClasses);
			else if(isRandomStateInitializers())
				stateProbs = initStateProbsRandom(numClasses, rand);
			else
				stateProbs = initStateProbsUniform(numClasses);
		}

		if (outputProbs == null)
		{
			outputProbs = initDiscreteOutputProbsRandom(numClasses, rand);
		}
		
		for (int i = 0; i < numClasses; i++)
		{
			estimators[i]=new DiscreteHMMEstimator(getNumStates(), getNumOutputs(), false);
			for (int s = 0; s < getNumStates(); s++)
				for (int o = 0; o < getNumOutputs(); o++)
				{
					estimators[i].addValue0(s, o, 100.0*state0Probs[i][s]*outputProbs[i][s][o]);
					for (int ps = 0; ps < getNumStates(); ps++)
						estimators[i].addValue(ps,s, o, 100.0*stateProbs[i][ps][s]*outputProbs[i][s][o]);
				}
		}
	}
	
	public void initEstimatorsMultivariateNormal(int numClasses, double state0Probs[][], double stateProbs[][][], DoubleVector outputMeans[][], Matrix outputVars[][], Instances data) throws Exception
	{
		estimators = new HMMEstimator[numClasses];
		
		// random initialization
		Random rand = new Random(getSeed());
		if (state0Probs == null)
		{
			if(isLeftRight())
				state0Probs = initState0ProbsLeftRight(numClasses);
			else if(isRandomStateInitializers())
				state0Probs = initState0ProbsRandom(numClasses, rand);
			else
				state0Probs = initState0ProbsUniform(numClasses);
		}

		if (stateProbs == null)
		{
			if(isLeftRight())
				stateProbs = initStateProbsLeftRight(numClasses);
			else if(isRandomStateInitializers())
				stateProbs = initStateProbsRandom(numClasses, rand);
			else
				stateProbs = initStateProbsUniform(numClasses);
		}
		
		for (int i = 0; i < numClasses; i++)
		{
			MultivariateNormalHMMEstimator est =new MultivariateNormalHMMEstimator(getNumStates(), false);
			estimators[i] = est;
			est.setCovarianceType(m_CovarianceType);
			est.setTied(isTied());
			est.setState0Probabilities(state0Probs[i]);
			est.setStateProbabilities(stateProbs[i]);
		}
		
		if(data == null)
		{
			initGaussianOutputProbsRandom(numClasses, outputMeans, outputVars);
		}
		else
		{
			if(isLeftRight())
				initGaussianOutputProbsAllData(numClasses, data, outputMeans, outputVars);
			else
				initGaussianOutputProbsCluster(numClasses, data, outputMeans, outputVars);
		}	
		
	}
	
	/*
	 * train an HMM classifier from data
	 * 
	 * parameter data adalah dataset yang akan digunakan untuk training
	 */
	public void buildClassifier(weka.core.Instances data) throws Exception {
		
		System.out.println("starting build classifier");
		// periksa apakah kita memiliki data kelas dan dalam bentuk yang benar
		if (data.classIndex() < 0)
		{
			System.err.println("tidak dapat menemukan index kelas");
			return;
		}
		if (!data.classAttribute().isNominal())
		{
			System.err.println("kelas atribut bukanlah nominal");
			return;
		}

		m_SeqAttr = -1;
		m_NumOutputs = 0;
		for(int i = 0; i < data.numAttributes(); i++)
		{
			Attribute attr = data.attribute(i);
			if(attr.isRelationValued())
			{
				if(attr.relation().attribute(0).isNominal())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
					m_NumOutputs  = attr.relation().numDistinctValues(0);
				}
				if(attr.relation().attribute(0).isNumeric())
				{
					m_SeqAttr = attr.index();
					assert(m_SeqAttr == i);
					this.setNumeric(true);
					m_NumOutputs  = -1;
					m_OutputDimension = attr.relation().numAttributes();
					break;
				}
				break;
			}
		}
		
		if (estimators == null)
			initEstimators(data.numClasses(), data);
		
		for(int i = 0; i < estimators.length; i++)
			System.out.println(i + " " + estimators[i]);
		
		if (m_SeqAttr < 0)
		{
			System.err.println("Tidak dapat menemukan atribut relasional yang sesuai dengan urutan tersebut");
			return;
		}
		
		if (data.numInstances() == 0)
		{
			System.err.println("No instances found");
			return;
		}
			
		double prevlik = -10000000.0;
		for (int step = 0; step < 100; step++)
		{
			double lik = EMStep(data);

			if(Math.abs((lik-prevlik)/lik) < getIterationCutoff())  
//
				break;
			prevlik = lik;
		}
		for(int i = 0; i < estimators.length; i++)
			System.out.println(i + " " + estimators[i]);
	}
	
	/*
	 * urutan sampel dari Model Markov Tersembunyi
	 *
	 * parameter numseqs adalah jumlah urutan dari sampel
	 * parameter length adalah panjang urutan tersebut
	 * return sebuah objek Instances yang berisi urutan
	 */
	public Instances sample(int numseqs, int length)
	{
		if(m_rand == null)
			m_rand = new Random(getSeed());
		return sample(numseqs, length, m_rand);
	}
	
	/*
	 * urutan sampel dari Model Markov Tersembunyi menggunakan generator nomor acak tertentu
	 *
	 * parameter numseqs adalah jumlah urutan sampel
	 * parameter length adalah panjang urutan tersebut
	 * parameter generator adalah generator nomor acak yang digunakan
	 * return sebuah objek Instances yang berisi urutan
	 */
	public Instances sample(int numseqs, int length, Random generator)
	{
		ArrayList<Attribute> attrs = new ArrayList<Attribute>();
		
		ArrayList<String> seqIds = new ArrayList<String>();
		for (int i = 0; i < numseqs; i++)
			seqIds.add("seq_"+i);
		attrs.add(new Attribute("seq-id", seqIds));
		
		ArrayList<String> classNames = new ArrayList<String>();
		for (int i = 0; i < estimators.length; i++)
			classNames.add("class_"+i);
		attrs.add(new Attribute("class", classNames));
		
		ArrayList<Attribute> seqAttrs = new ArrayList<Attribute>();
		if(isNumeric())
		{
			for(int i = 0; i < getOutputDimension(); i++)
			{
				seqAttrs.add(new Attribute("output_"+i));
			}
		}
		else
		{
			ArrayList<String> outputs = new ArrayList<String>();
			for(int i = 0; i < getNumOutputs(); i++)
				outputs.add("output_"+i);
			seqAttrs.add(new Attribute("output", outputs));
		}
		Instances seqHeader = new Instances("seq", seqAttrs, 0);
		attrs.add(new Attribute("sequence", seqHeader));
		
		Instances seqs = new Instances("test", attrs, numseqs);
		seqs.setClassIndex(1);
		
		for (int seq=0; seq<numseqs; seq++)
		{

			seqs.add(new DenseInstance(3));
			Instance inst = seqs.lastInstance();
			inst.setValue(0, seqIds.get(seq));
			int classId = m_rand.nextInt(classNames.size());
			inst.setValue(1, classNames.get(classId));
			//System.out.print("class "+classId+":");
			
			HMMEstimator est = estimators[classId];
			
			Instances sequence = new Instances(seqIds.get(seq), seqAttrs, length);
			int state = est.Sample0(sequence, generator);
			for (int i = 1; i < length; i++)
			{

				state = est.Sample(sequence, state, generator);
			}
			Attribute seqA = seqs.attribute(2);
			inst.setValue(seqA, seqA.addRelation(sequence));
		}
			
		return seqs;
	}
	
	public static void main(String [] argv) {
	    runClassifier(new HMM(), argv);
	  }
}
