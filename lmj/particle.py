# Copyright (c) 2010 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''A simple particle filter implementation.'''

# Much of the inspiration for this code came from the small sample at
# http://www.scipy.org/Cookbook/ParticleFilter.
#
# http://homepages.inf.ed.ac.uk/mdewar1/python_tutorial/bootstrap.html enhanced
# the implementation by bridging it a bit with the literature.

import numpy
import logging
from numpy import random as rng
from itertools import izip as zip


def uniform_displacement(temperature):
    '''Return a callable that displaces the particles in a particle filter.

    This particular model simply displaces the particles in a filter by a
    uniformly distributed random value between -temperature and temperature.
    '''
    def displace(particles):
        particles += rng.uniform(-temperature, temperature, particles.shape)
    return displace


def euclidean_distance(particles, observation):
    '''Return a matrix with the euclidean distance from an observation.

    particles: A two-dimensional numpy array of particles that represent the
      current estimate of a distribution.
    observation: A one-dimensional numpy array containing a new observation.
    '''
    delta = particles - observation
    return numpy.sqrt((delta * delta).sum(axis=1))


def euclidean_assessment(weights, particles, observation):
    '''Return the inverse Euclidean distance between particles and observation.

    weights: A vector of weights, one weight for each particle.
    particles: A two-dimensional array of particles that represent the current
      estimate of a distribution.
    observation: A vector containing a new observation.
    '''
    weights += 1.0 / (1.0 + euclidean_distance(particles, observation))


def log_euclidean_assessment(weights, particles, observation):
    '''Return the inverse Euclidean distance between particles and observation.

    weights: A vector of weights, one weight for each particle.
    particles: A two-dimensional array of particles that represent the current
      estimate of a distribution.
    observation: A vector containing a new observation.
    '''
    z = numpy.log(1.0 + euclidean_distance(particles, observation))
    weights += 1.0 / (1.0 + z)


def neighborhood_euclidean_assessment(proportion):
    '''Return a callable that reweights particles in a neighborhood.

    proportion: A fraction in [0, 1] that determines the proportion of particles
      that will be reweighted.
    '''
    def calculate(weights, particles, observation):
        p = len(particles)
        r = int(p * proportion)
        # find the r closest particles to the observation.
        distances = euclidean_distance(particles, observation)
        z = distances.argsort()[:r]
        # alter just the weights of these particles. the probability of the
        # update must sum to proportion, so that the probability of the whole
        # weight vector still sums to 1.
        w = (1.0 / (1.0 + distances[z]))
        weights[z] = proportion * w / w.sum()
    return calculate


def resample_iteratively(iterations):
    '''Resample every N iterations.'''
    def test(w, state=dict(count=0)):
        state['count'] += 1
        return 0 == state['count'] % iterations
    return test


def resample_randomly(rate):
    '''Resample randomly at some constant, uniform rate.'''
    return lambda w: rng.uniform() < rate


def resample_when_skewed(threshold):
    '''Resample when the weights in a particle filter are too skewed.

    threshold: A numeric threshold that determines the tolerated skew above
      which particles should be resampled. If this is around 2, resampling
      occurs whenever about 25 % of the particles have about 75 % of the weight.
      Other easily interpreted thresholds are :

        50/50  1.0000 (i.e. resample wih every observation)
        60/40  1.1667
        70/30  1.7619
        80/20  3.2500
        90/10  8.1111
    '''
    return lambda w: (len(w) * (w * w).sum()) > threshold


class Filter(object):
    '''A particle filter is a discrete estimate of a probability distribution.

    The filter implementation here maintains a small-ish set of discrete
    particles to represent an estimate of a probability distribution that
    evolves according to some unobserved dynamics.

    Each particle is positioned somewhere in the space being observed, and is
    explicitly weighted with an estimate of the likelihood of that particle.

    The weighted sum of the particles can be used as the expectation of the
    underlying distribution, or---more generally---the weighted particles may
    be used to calculate the expected value of any function under this
    distribution.
    '''

    def __init__(self,
                 particles,
                 dimensions=1,
                 displace_particles=None,
                 assess_particles=None,
                 resample_test=None):
        '''Initialize this filter.

        particles: The number of particles to use. Smaller values result in
          lower accuracy but require less computational effort.
        dimensions: The dimensionality of the space from which we draw
          observations.
        displace_particles: A callable that takes one argument---the current
          particles---and displaces them according to the system being modeled.
        assess_particles: A callable that takes three arguments---the current
          weights, the current particles, and an observation---and modifies
          the weights according to some assessment procedure.
        resample_test: A callable that takes one argument---the current weights
          in the filter---and returns a boolean indicating whether to resample
          the particles.
        '''
        self._particles = rng.randn(particles, dimensions)
        self._weights = numpy.ones((particles, ), 'd') / particles
        self._cuml_weights = None # cache the cumulative weight distribution
        self._displace = displace_particles or uniform_displacement(1)
        self._assess = assess_particles or euclidean_assessment
        self._resample_test = resample_test or resample_when_skewed(2)

    def expectation(self):
        '''Get the expected value of the identity function under our filter.'''
        w = self._weights.reshape(self._weights.shape + (1, ))
        return (self._particles * w).sum(axis=0)

    def sample(self):
        '''Return a sample from our filter's distribution.'''
        if self._cuml_weights is None:
            self._cuml_weights = self._weights.cumsum()
        index = numpy.searchsorted(self._cuml_weights, rng.random())
        return self._particles[index]

    def _resample(self):
        '''Resample the particles in our filter using the current weights.'''
        logging.debug('resampling')
        if self._cuml_weights is None:
            self._cuml_weights = self._weights.cumsum()
        p = self._particles.shape[0]
        indices = numpy.searchsorted(self._cuml_weights, rng.random(p))
        self._particles = self._particles[indices]
        self._weights[:] = 1.0 / p
        self._cuml_weights = None

    def observe(self, observation):
        '''Update the filter based on a new observation.

        observation: A numpy array containing a new observation.
        '''
        self._displace(self._particles)
        self._assess(self._weights, self._particles, observation)
        self._weights /= self._weights.sum()
        self._cuml_weights = None
        if self._resample_test(self._weights):
            self._resample()

    def iterparticles(self):
        '''Iterate over the particles and weights in this filter.'''
        for i in xrange(len(self._particles)):
            yield self._particles[i], self._weights[i]


class ChannelFilter(object):
    '''A particle filter for multi-channel data.'''

    def __init__(self,
                 channels,
                 particles,
                 dimensions=1,
                 displace_particles=None,
                 assess_particles=None,
                 resample_test=None):
        '''Initialize this filter.

        channels: The number of channels of data that we receive in a frame.
        particles: The number of particles to use for modeling each channel.
          Smaller values result in lower accuracy but require less computational
          effort.
        dimensions: The dimensionality of the space from which we draw
          observations in each channel.
        displace_particles: A callable that takes one argument---the current
          particles---and displaces them according to the system being modeled.
        assess_particles: A callable that takes two arguments---the current
          particles and an observation---and returns an array of weights to be
          normalized and mixed with the current weights.
        resample_test: A callable that takes one argument---the current weights
          in the filter---and returns a boolean indicating whether to resample
          the particles.
        '''
        self._particles = rng.randn(channels, particles, dimensions)
        self._weights = numpy.ones((channels, particles), 'd') / particles
        self._cuml_weights = [None] * channels
        self._displace = displace_particles or uniform_displacement(1)
        self._assess = assess_particles or euclidean_assessment
        self._resample_test = resample_test or resample_when_skewed(2)

    def expectations(self):
        '''Get the expected value of the identity function under our filter.'''
        w = self._weights.reshape(self._weights.shape + (1, ))
        return (self._particles * w).sum(axis=1)

    def samples(self):
        '''Return a sample from each channel in our filter's distributions.'''
        channels, _, dimensions = self._particles.shape
        sample = numpy.empty((channels, dimensions), 'd')
        for c in xrange(channels):
            if self._cuml_weights[c] is None:
                self._cuml_weights[c] = self._weights[c].cumsum()
            index = numpy.searchsorted(self._cuml_weights[c], rng.random())
            sample[c] = self._particles[c, index]
        return sample

    def _resample(self, c):
        '''Resample the particles in a specific channel.'''
        logging.debug('resampling in channel %d', c)
        if self._cuml_weights[c] is None:
            self._cuml_weights[c] = self._weights[c].cumsum()
        p = self._particles.shape[1]
        indices = numpy.searchsorted(self._cuml_weights[c], rng.random(p))
        self._particles[c] = self._particles[c, indices]
        self._weights[c, :] = 1.0 / p
        self._cuml_weights[c] = None

    def observe(self, observations):
        '''Update the filter based on a new observation.

        observations: A numpy array containing a new observation for each
          channel. This array must be shaped as (channels, dimensions).
        '''
        self._displace(self._particles)
        channels = zip(self._weights, self._particles, observations)
        for channel, (weights, particles, obs) in enumerate(channels):
            self._assess(weights, particles, obs)
            weights /= weights.sum()
            self._cuml_weights[channel] = None
            if self._resample_test(weights):
                self._resample(channel)

    def iterparticles(self, channel):
        '''Iterate over the particles and weights in this filter.'''
        for i in xrange(len(self._particles[channel])):
            yield self._particles[channel, i], self._weights[channel, i]
