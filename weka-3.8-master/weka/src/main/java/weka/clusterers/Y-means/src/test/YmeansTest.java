
package weka.clusterers;

import weka.clusterers.AbstractClustererTest;
import weka.clusterers.Clusterer;

import junit.framework.Test;
import junit.framework.TestSuite;


public class Y_meansTest 
	extends AbstractClustererTest {

	public Y_meansTest(String name) { 
		super(name);  
	}

	public Clusterer getClusterer() {
		return new Y_means();
	}

	public static Test suite() {
		return new TestSuite(Y_meansTest.class);
	}

	public static void main(String[] args){
		junit.textui.TestRunner.run(suite());
	}
}
