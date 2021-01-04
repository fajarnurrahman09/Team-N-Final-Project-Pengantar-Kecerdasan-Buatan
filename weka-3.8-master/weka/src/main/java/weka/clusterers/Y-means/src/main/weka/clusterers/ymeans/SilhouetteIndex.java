
package weka.clusterers.ymeans;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Locale;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.clusterers.AbstractClusterer;

public class SilhouetteIndex implements Serializable, ClusterEvaluator {


	static final long serialVersionUID = -305533168492651330L;


	protected ArrayList<Double> m_clustersSilhouette;


	protected double m_globalSilhouette;

	public SilhouetteIndex() {
		m_clustersSilhouette = new ArrayList<Double>();
		m_globalSilhouette = 0.0;
	}


	@SuppressWarnings("unchecked")
	public void evaluate(AbstractClusterer clusterer, Instances centroids,
		Instances instances, DistanceFunction distanceFunction) throws Exception {

		if (clusterer == null || instances == null)
			throw new Exception("SilhouetteIndex: the clusterer or instances are null!");


		ArrayList<Instance>[] clusteredInstances =
			(ArrayList<Instance>[]) new ArrayList<?>[centroids.size()];


		for (int i = 0; i < centroids.size(); i++)
			clusteredInstances[i] = new ArrayList<Instance>();

		for (int i = 0; i < instances.size(); i++)
			clusteredInstances[ clusterer.clusterInstance( instances.get(i) ) ]
				.add( instances.get(i) );


		for (int i = 0; i < clusteredInstances.length; i++) {
			double centroidSilhouetteIndex = 0.0;

			for (int j = 0; j < clusteredInstances[i].size(); j++) {
				double pointSilhouetteIndex = 0.0;
				double meanDistSameC  = 0.0;
				double meanDistOtherC = 0.0;


				Instance i1 = clusteredInstances[i].get(j);

				for (int k = 0; k < clusteredInstances[i].size(); k++) {
					
					if (k == j)
						continue;

					
					Instance i2 = clusteredInstances[i].get(k);
					meanDistSameC += distanceFunction.distance(i1, i2);
				}

			
				meanDistSameC /= (clusteredInstances[i].size() - 1);

				
				double minDistance = Double.MAX_VALUE;
				int minCentroid = 0;

				for (int k = 0; k < centroids.size(); k++) {
					
					if (k == i)
						continue;

					
					Instance i2 = centroids.get(k);
					double distance = distanceFunction.distance(i1, i2);

					
					if (distance < minDistance) {
						minDistance = distance;
						minCentroid = k;
					}
				}

				for (int k = 0; k < clusteredInstances[minCentroid].size(); k++) {

					Instance i2 = clusteredInstances[minCentroid].get(k);


					meanDistOtherC += distanceFunction.distance(i1, i2);
				}

			
				meanDistOtherC /= (clusteredInstances[minCentroid].size() - 1);

				pointSilhouetteIndex = (meanDistOtherC - meanDistSameC) / 
					Math.max( meanDistSameC, meanDistOtherC );

				centroidSilhouetteIndex += pointSilhouetteIndex;
			}

			centroidSilhouetteIndex /= (clusteredInstances[i].size() - 1);
			m_globalSilhouette += centroidSilhouetteIndex;

			m_clustersSilhouette.add( centroidSilhouetteIndex );
		}

		m_globalSilhouette /= m_clustersSilhouette.size();
	}


	public ArrayList<Double> getClustersSilhouette() {
		return m_clustersSilhouette;
	}


	public double getGlobalSilhouette() {
		return m_globalSilhouette;
	}


	public String evalSilhouette(double si) {
		String eval = "";

		if (si > 0.70)
			eval = "strong structure!";
		else if (si >  0.50 && si <= 0.70)
			eval = "reasonably structure!";
		else if (si >  0.25 && si <= 0.50)
			eval = "weak structure!";
		else if (si <= 0.25)
			eval = "a non substancial structure was found!";

		return eval;
	}

	 @Override
	 public String toString() {
	 	StringBuffer description = new StringBuffer("");

		/* Clusters. */
		for (int i = 0; i < m_clustersSilhouette.size(); i++) {
			double si = m_clustersSilhouette.get(i);
			description.append("   Cluster " + i + ": " + String.format(Locale.US, "%.4f", si)
				+ ", veredict: " + evalSilhouette(si) + "\n");
		}

		description.append("   Mean: " + String.format(Locale.US, "%.4f", m_globalSilhouette)
			+ ", veredict: " + evalSilhouette(m_globalSilhouette));

		return description.toString();
	 }
}
