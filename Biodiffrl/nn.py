import jax
import jax.numpy as jp
import flax
import flax.linen as nn
import optax


class Controller_NN(nn.Module):
    
    in_dims: int
    out_dims: int
    
    def setup(self):
        # Features means the output dimension
        self.linear1 = nn.Dense(features=512)
        self.linear2 = nn.Dense(features=256)
        self.linear3 = nn.Dense(features=512)
        self.linear4 = nn.Dense(features=512)
        # The last layer will output the mean and logstd
        self.linear5 = nn.Dense(features=self.out_dims*2)
        
    
    def __call__(self, x, key):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        x = nn.relu(x)
        x = self.linear3(x)
        x = nn.relu(x)
        # x = self.linear4(x)
        # x = nn.relu(x)
        x = self.linear5(x)
        # The last layer of the neural requires samping
        mean = x[:self.out_dims]
        logstd = x[self.out_dims:]
        std = jp.exp(logstd)
        # samples = jp.clip(jax.random.normal(key)*std*0.3 + mean, -3, 3)
        sample = nn.relu(jax.random.normal(key)*std*0.2 + mean)
        mean = nn.relu(mean)
        return sample, mean, logstd
    
    def init_parameters(self, key):
        # Init the model
        sub_key = jax.random.split(key,1)[0]
        # The second parameter is the dommy input
        params = self.init(key, jp.empty([1, self.in_dims]), sub_key)
        return params, sub_key
    
    def get_fn(self):
        return lambda params, states, key : self.apply(params, states, key)
    
class Critic_NN(nn.Module):

    in_dims: int
    out_dims: int
    
    def setup(self):
        # Features means the output dimension
        self.linear1 = nn.Dense(features=512)
        self.linear2 = nn.Dense(features=1024)
        self.linear3 = nn.Dense(features=512)
        self.linear4 = nn.Dense(features=256)
        # The last layer will output the mean and logstd
        self.linear5 = nn.Dense(features=self.out_dims)
        
    
    def __call__(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        x = nn.relu(x)
        x = self.linear3(x)
        x = nn.relu(x)
        x = self.linear4(x)
        x = nn.relu(x)
        x = self.linear5(x)
        # The last layer of the neural requires samping
        return -nn.relu(x)
        # return x
    
    def init_parameters(self, key):
        # Init the model
        sub_key = jax.random.split(key,1)[0]
        # The second parameter is the dommy input
        params = self.init(key, jp.empty([1, self.in_dims]))
        return params, sub_key
    
    def get_fn(self):
        return lambda params, states: self.apply(params, states)