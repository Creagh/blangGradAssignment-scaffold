package matchings;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import bayonet.distributions.Random;
import blang.core.LogScaleFactor;
import blang.distributions.Generators;
import blang.mcmc.ConnectedFactor;
import blang.mcmc.SampledVariable;
import blang.mcmc.Sampler;
import briefj.collections.UnorderedPair;

/**
 * Each time a Permutation is encountered in a Blang model, 
 * this sampler will be instantiated. 
 */
public class PermutationSampler implements Sampler {
  /**
   * This field will be populated automatically with the 
   * permutation being sampled. 
   */
  @SampledVariable Permutation permutation;
  /**
   * This will contain all the elements of the prior or likelihood 
   * (collectively, factors), that depend on the permutation being 
   * resampled. 
   */
  @ConnectedFactor List<LogScaleFactor> numericFactors;

  @Override
  public void execute(Random rand) {
    // Fill this.
	
	  /** 
	   * Metropolis-Hastings Algorithm for sampling Permutations 
	   * (equivalently, perfect bipartite matchings).
	   */
	  
	  List<Integer> oldPerm; // old permutation
	  double oldDens; // log density of old permutation
	  double newDens; // log density of new permutation
	  double alpha; // the acceptance probability
	  boolean bernoulli; // a Bernoulli RV
	  
	  /** 
	   * Store the old permutation and calculate log density
	   */
	  oldPerm = new ArrayList<Integer>(permutation.getConnections());
	  oldDens = logDensity();
	  
	  /**
	   *  Sample a new permutation and calculate new log density.
	   *  New permutation is proposed by sampling a permutation
	   *  uniformly at random, using a shuffle in place method.
	   */
	  permutation.sampleUniform(rand);
	  newDens = logDensity();
	  
	  /**
	   *  Calculate the acceptance probability
	   */
	  alpha = Math.min(1, Math.exp(newDens - oldDens));
	  
	  /**
	   *  Determine whether to accept the new permutation
	   *  Flip a weighted coin: heads = accept, tails = reject
	   */
	  bernoulli = Generators.bernoulli(rand, alpha);
	  
	  if(!bernoulli) {
		  // Reject & replace all new values with the old permutation
		  for(int i=0; i < oldPerm.size(); i++) {
			  permutation.getConnections().set(i, oldPerm.get(i));
		  }
	  }
	  
  }
  
  private double logDensity() {
    double sum = 0.0;
    for (LogScaleFactor f : numericFactors)
      sum += f.logDensity();
    return sum;
  }
}
