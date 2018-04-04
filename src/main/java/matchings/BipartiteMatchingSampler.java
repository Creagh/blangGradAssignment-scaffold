package matchings;

import java.util.ArrayList;
import java.util.List;

import bayonet.distributions.Multinomial;
import bayonet.distributions.Random;
import blang.core.LogScaleFactor;
import blang.distributions.Generators;
import blang.mcmc.ConnectedFactor;
import blang.mcmc.SampledVariable;
import blang.mcmc.Sampler;

/**
 * Each time a Permutation is encountered in a Blang model, 
 * this sampler will be instantiated. 
 */
public class BipartiteMatchingSampler implements Sampler {
  /**
   * This field will be populated automatically with the 
   * permutation being sampled. 
   */
  @SampledVariable BipartiteMatching matching;
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
	   * Metropolis-Hastings Algorithm for sampling Bipartite Matchings 
	   */
	  
	  List<Integer> oldMatch; // old matching
	  double oldDens; // log density of old matching
	  double newDens; // log density of new matching
	  double alpha; // the acceptance probability
	  boolean bernoulli; // a Bernoulli RV
	  
	  /** 
	   * Store the old matching and calculate log density
	   */
	  oldMatch = new ArrayList<Integer>(matching.getConnections());
	  oldDens = logDensity();
	  
	  /**
	   *  Sample a new matching and calculate new log density.
	   *  New matching is proposed by sampling a matching
	   *  uniformly at random---see BipartiteMatching.xtend
	   */
	  matching.sampleUniform(rand);
	  newDens = logDensity();
	  
	  /**
	   *  Calculate the acceptance probability
	   */
	  alpha = Math.min(1, Math.exp(newDens - oldDens));
	  
	  /**
	   *  Determine whether to accept the new matching
	   *  Flip a weighted coin: heads = accept, tails = reject
	   */
	  bernoulli = Generators.bernoulli(rand, alpha);
	  
	  if(!bernoulli) {
		  // Reject & replace all new values with the old matching
		  for(int i=0; i < oldMatch.size(); i++) {
			  matching.getConnections().set(i, oldMatch.get(i));
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
