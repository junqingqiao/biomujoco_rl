from dataclasses import dataclass
import functools
import jax
import jax.numpy as jp


# Register the Experience class as pytree
@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=['states', 'next_states', 'actions', 'rewards'],
                   meta_fields=[])
@dataclass
class experience(object):
    states : jax.Array
    next_states : jax.Array
    actions : jax.Array
    rewards : jax.Array
    # dones: jax.Array
    
    # def __init__(self, state, next_state, action, reward):
    #     self.state = state
    #     self.next_state = next_state
    #     self.action = action
    #     self.reward = reward



@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=['memory_size', 'state_num', 'action_num', 'reward_num'],
                   meta_fields=[])
@dataclass(frozen=True)
class memory_settings():
    memory_size  : jax.Array
    state_num    : jax.Array
    action_num   : jax.Array
    reward_num   : jax.Array


class memory(): 
    
    @staticmethod
    def append_fn(exp_item, exp_pool_item):
        return jp.concatenate([exp_item, exp_pool_item],axis=0)
    
    @staticmethod
    def trim_fn(exp_pool_item, length):
        return exp_pool_item[:length]
    
    @staticmethod
    def sample_fn(exp_pool_item, index):
        return exp_pool_item[index]
    
    staticmethod
    @functools.partial(jax.jit, static_argnames="settings")
    def add_exp(settings:memory_settings, exp_pool:experience, exp:experience):
        exp.states = jp.reshape(exp.states,(-1, settings.state_num))
        exp.next_states = jp.reshape(exp.next_states,(-1, settings.state_num))
        exp.actions = jp.reshape(exp.actions, (-1, settings.action_num))
        exp.rewards = jp.reshape(exp.rewards, (-1, settings.reward_num))
        
        if(exp_pool == None):
            exp_pool = exp
        else:
            exp_pool = jax.tree.map(memory.append_fn, exp, exp_pool)
            
        len = exp_pool.states.shape[0]
        
        # Forget the outdated memory
        if(len> settings.memory_size):
            exp_pool = jax.tree.map(lambda x: memory.trim_fn(x, settings.memory_size), exp_pool)
        
        return exp_pool
        
    @staticmethod
    @functools.partial(jax.jit, static_argnames="batch_size")
    def sample(exp_pool, batch_size, key)->experience :
        index = jax.random.choice(
            key,
            jp.arange(exp_pool.states.shape[0]),
            shape = (batch_size,),
            replace=False
        )
        return jax.tree.map(lambda x: memory.sample_fn(x, index), exp_pool)