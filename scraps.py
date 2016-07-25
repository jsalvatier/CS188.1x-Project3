"""
AGENT:
	s, internal_state --> a, query

Environment
	s, a --> s', r(s,a)
"""


class MDP(object):
	def __init__(self, states, actions, T, R, discount):
		self.__dict__.update(locals())

class Agent(object):
	def __init__(self, states, actions, policy):
		self.__dict__.update(locals())

class ActiveAgent(object):
	def __init__(self, states, actions, policy, query_fn):
		self.__dict__.update(locals())



assert agent.states == mdp.states
assert agent.actions == mdp.actions




"""
Implementation plans:


ActiveLearningAgent:
	Returns a binary query action, as well as an MDP action.
	The Query action can be decided by a separate algorithm, or not.


LearningAgent_ACTIVATOR:
	Takes a ValueEstimationAgent and returns an ActiveLearningAgent
	Encodes an active learning algorithm that is based


Environment_ACTIVATOR:
	Takes an environment and adds the active learning components.

"""
