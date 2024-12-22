import numpy as np

class MarkovChain(object):
    def __init__(self, transition_prob):
        """
        transition_prob: dict
        A dict object representing the transition probabilities in markov chain. Should be of the form:
        {'state1': {'state1': 0.1, 'state2': 0.4}, 'state2': {...} }

        """
        self.transition_prob = transition_prob
        self.states = list(transition_prob.keys())
        

    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time instance
        
        it takes current state and randomly picks the next one
        
        """
        return np.random.choice(self.states, p=[self.transition_prob[current_state][next_state] for next_state in self.states])
    
    def generate_states(self, current_state, n = 10):
        future_states = []
        for i in range(n):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states
    


# EXAMPLE:

transition_prob = {'Sunny': {'Sunny': 0.8, 'Rainy': 0.19, 'Snowy': 0.01}, 'Rainy': {'Sunny': 0.2, 'Rainy': 0.7, 'Snowy': 0.1}, 'Snowy': {'Sunny': 0.1, 'Rainy': 0.2, 'Snowy': 0.7}}
weather_chain = MarkovChain(transition_prob=transition_prob)

print("\n --------------------------------------------------------------------- \n")
print(f"\n Next weather, given current weather is Sunny: {weather_chain.next_state(current_state='Sunny')}")
print(f" \n Next weather, given current weather is Rainy: {weather_chain.next_state(current_state='Rainy')}")

print(f"\n Next 10 weather states when current weather is Snowy: {weather_chain.generate_states(current_state='Snowy', n=10)}")

print("\n --------------------------------------------------------------------- \n")
