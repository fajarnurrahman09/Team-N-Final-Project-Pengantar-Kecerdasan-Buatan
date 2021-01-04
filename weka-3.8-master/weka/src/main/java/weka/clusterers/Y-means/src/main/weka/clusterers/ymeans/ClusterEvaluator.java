
package weka.clusterers.ymeans;

import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.clusterers.AbstractClusterer;

public interface ClusterEvaluator {

	void evaluate(AbstractClusterer clusterer, Instances centroids,
		Instances instances, DistanceFunction distanceFunction) throws Exception;
}
