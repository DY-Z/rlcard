
import os
import threading
import time
import timeit
import pprint
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn

import logging
import traceback

from rlcard.agents.dmc_agent.file_writer import FileWriter
from rlcard.agents.dmc_agent.model import DMCModel
from rlcard.agents.dmc_agent.utils import get_batch, create_buffers,  act,  log


'''def act(i, device, T, free_queue, full_queue, model, buffers, env):
    try:
        print("act start")
        log.info('Device %i Actor %i started.', device, i)
        torch_device = torch.device('cuda:'+str(device))

        # Configure environment
        env.seed(i)
        env.set_agents(model.get_agents())

        done_buf = [[]]
        episode_return_buf = [[]]
        target_buf = [[]]
        state_buf = [[]]
        action_buf = [[]]
        size = [0]

        print("loop start")
        while True:
            print("another iter")
            trajectories, payoffs = env.run(is_training=True)
            p = 0
            size[p] += len(trajectories[p][:-1]) // 2
            diff = size[p] - len(target_buf[p])
            if diff > 0:
              done_buf[p].extend([False for _ in range(diff-1)])
              done_buf[p].append(True)
              episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
              episode_return_buf[p].append(float(payoffs[p]))
              target_buf[p].extend([float(payoffs[p]) for _ in range(diff)])
            # State and action
              for i in range(0, len(trajectories[p])-2, 2):
                state = trajectories[p][i]['obs']
                action = env.get_action_feature(trajectories[p][i+1])
                state_buf[p].append(torch.from_numpy(state))
                action_buf[p].append(torch.from_numpy(action))
                
            while size[p] > T:
                index = free_queue[p].get()
                if index is None:
                    break
                for t in range(T):
                    buffers[p]['done'][index][t, ...] = done_buf[p][t]
                    buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                    buffers[p]['target'][index][t, ...] = target_buf[p][t]
                    buffers[p]['state'][index][t, ...] = state_buf[p][t]
                    buffers[p]['action'][index][t, ...] = action_buf[p][t]
                full_queue[p].put(index)
                done_buf[p] = done_buf[p][T:]
                episode_return_buf[p] = episode_return_buf[p][T:]
                target_buf[p] = target_buf[p][T:]
                state_buf[p] = state_buf[p][T:]
                action_buf[p] = action_buf[p][T:]
                size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
'''

def create_optimizers(trainable_posi, learning_rate, momentum, epsilon, alpha, learner_model):
    optimizers = []
    for player_id in range(len(trainable_posi)):
      if trainable_posi[player_id] == 1:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(player_id),
            lr=learning_rate,
            momentum=momentum,
            eps=epsilon,
            alpha=alpha)
      else:
        optimizer = 0

      optimizers.append(optimizer)
    return optimizers

def compute_loss(logits, targets):
    loss = ((logits - targets)**2).mean()
    return loss

def learn(position,
          actor_models,
          agent,
          batch,
          optimizer,
          training_device,
          max_grad_norm,
          mean_episode_return_buf,
          lock):
    """Performs a learning (optimization) step."""
    device = torch.device('cuda:'+str(training_device))
    state = torch.flatten(batch['state'].to(device), 0, 1).float()
    action = torch.flatten(batch['action'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    print("Called")

    with lock:
        values = agent.forward(state, action)
        loss = compute_loss(values, target)
        stats = {
            'mean_episode_return_'+str(position): torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_'+str(position): loss.item(),
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        for actor_model in actor_models:
            actor_model.get_agent(position).load_state_dict(agent.state_dict())
        return stats


class DMCTrainer:    
    def __init__(self,
                 env,
                 trainable_posi = [1,0,0,0],
                 load_model=False,
                 xpid='dmc',
                 save_interval=30,
                 num_actor_devices=1,
                 num_actors = 1,
                 training_device=0,
                 savedir='experiments/dmc_result',
                 total_frames=100000000000,
                 exp_epsilon=0.01,
                 batch_size=32,
                 unroll_length=100,
                 num_buffers=50,
                 num_threads=1,
                 max_grad_norm=40,
                 learning_rate=0.001,
                 alpha=0.99,
                 momentum=0.1,
                 epsilon=0.00001):
        '''
        Deep Monte-Carlo

        Args:
            env: RLCard environment
            load_model (boolean): Whether loading an existing model
            xpid (string): Experiment id (default: dmc)
            save_interval (int): Time interval (in minutes) at which to save the model
            num_actor_devices (int): The number devices used for simulation
            num_actors (int): Number of actors for each simulation device
            training_device (int): The index of the GPU used for training models
            savedir (string): Root dir where experiment data will be saved
            total_frames (int): Total environment frames to train for
            exp_epsilon (float): The prbability for exploration
            batch_size (int): Learner batch size
            unroll_length (int): The unroll length (time dimension)
            num_buffers (int): Number of shared-memory buffers
            num_threads (int): Number learner threads
            max_grad_norm (int): Max norm of gradients
            learning_rate (float): Learning rate
            alpha (float): RMSProp smoothing constant
            momentum (float): RMSProp momentum
            epsilon (float): RMSProp epsilon
        '''
        self.env = env

        self.plogger = FileWriter(
            xpid=xpid,
            rootdir=savedir,
        )

        self.checkpointpath = os.path.expandvars(
            os.path.expanduser('%s/%s/%s' % (savedir, xpid, 'model.tar')))

        self.T = unroll_length
        self.B = batch_size

        self.xpid = xpid
        self.load_model = load_model
        self.savedir = savedir
        self.save_interval = save_interval
        self.num_actor_devices = num_actor_devices
        self.num_actors = num_actors
        self.training_device = training_device
        self.total_frames = total_frames
        self.exp_epsilon = exp_epsilon
        self.num_buffers = num_buffers
        self.num_threads = num_threads
        self.max_grad_norm = max_grad_norm
        self.learning_rate =learning_rate
        self.alpha = alpha
        self.momentum = momentum
        self.epsilon = epsilon

        self.trainable_posi = trainable_posi
        self.total_trainable = sum(trainable_posi)

        self.action_shape = self.env.action_shape
        if self.action_shape[0] == None:  # One-hot encoding
            self.action_shape = [[self.env.num_actions] for _ in range(self.env.num_players)]

        self.mean_episode_return_buf = [deque(maxlen=100) for _ in range(self.env.num_players)]
        '''self.mean__episode_return_buf = []
        for p in range(len(trainable_posi)):
          if trainable_posi[p] == 1:
            self.mean__episode_return_buf.append(deque(maxlen=100))
          else:
            self.mean__episode_return_buf.append([])'''

    def start(self):
        # Initialize actor models
        models = []
        for device in range(self.num_actor_devices):
            model = DMCModel(self.trainable_posi,
                      self.env.state_shape,
                      self.action_shape,
                      exp_epsilon=self.exp_epsilon,
                      device=device)
            
            model.share_memory()
            model.eval()
            models.append(model)

        # Initialize buffers
        buffers = create_buffers(self.T,
                                 self.num_buffers,
                                 self.env.state_shape,
                                 self.action_shape)

        # Initialize queues
        actor_processes = []
        ctx = mp.get_context('spawn')
        free_queue = []
        full_queue = []
        print("reach full queue")
        for device in range(self.num_actor_devices):
            _free_queue = [ctx.SimpleQueue() for _ in range(self.env.num_players)]
            _full_queue = [ctx.SimpleQueue() for _ in range(self.env.num_players)]
            free_queue.append(_free_queue)
            full_queue.append(_full_queue)

        print("finished full queue")
        # Learner model for training
        learner_model = DMCModel(self.trainable_posi,
                      self.env.state_shape,
                      self.action_shape,
                      device=self.training_device)
        
        print(learner_model.agents)

        # Create optimizers
        optimizers = create_optimizers(self.trainable_posi,
                                       self.learning_rate,
                                       self.momentum,
                                       self.epsilon,
                                       self.alpha,
                                       learner_model)

        # Stat Keys
        stat_keys = []
        for p in range(self.env.num_players):
          if self.trainable_posi[p] == 1:
            stat_keys.append('mean_episode_return_'+str(p))
            stat_keys.append('loss_'+str(p))
        frames, stats = 0, {k: 0 for k in stat_keys}

        # Load models if any
        if self.load_model and os.path.exists(self.checkpointpath):
            checkpoint_states = torch.load(
                    self.checkpointpath, map_location="cuda:"+str(self.training_device)
            )
            for p in range(self.env.num_players):
              if self.trainable_posi[p] == 1:
                learner_model.get_agent(p).load_state_dict(checkpoint_states["model_state_dict"][p])
                optimizers[p].load_state_dict(checkpoint_states["optimizer_state_dict"][p])
                for device in range(self.num_actor_devices):
                    models[device].get_agent(p).load_state_dict(learner_model.get_agent(p).state_dict())
            stats = checkpoint_states["stats"]
            frames = checkpoint_states["frames"]
            log.info(f"Resuming preempted job, current stats:\n{stats}")

        print("done load model")
        # Starting actor processes
        for device in range(self.num_actor_devices):
            num_actors = self.num_actors
            for i in range(self.num_actors):
                print(i, "reach")
                actor = ctx.Process(
                    target=act,
                    args=(i, device, self.T, free_queue[device], full_queue[device], models[device], buffers[device], self.env))
                print("actor done")
                print(actor)
                actor.start()
                print("finish start")
                actor_processes.append(actor)
                print(i, "done")

        print("done actor process")
        def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
            """Thread target for the learning process."""
            nonlocal frames, stats
            while frames < self.total_frames:
                batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position], self.B, local_lock)
                _stats = learn(position, models, learner_model.get_agent(position), batch,
                    optimizers[position], self.training_device, self.max_grad_norm, self.mean_episode_return_buf, position_lock)

                with lock:
                    for k in _stats:
                        stats[k] = _stats[k]
                    to_log = dict(frames=frames)
                    to_log.update({k: stats[k] for k in stat_keys})
                    self.plogger.log(to_log)
                    frames += self.T * self.B

        for device in range(self.num_actor_devices):
            for m in range(self.num_buffers):
                for p in range(self.env.num_players):
                  if self.trainable_posi[p] == 1:
                    free_queue[device][p].put(m)

        threads = [] 
        locks = [[threading.Lock() if self.trainable_posi[i] == 1 else 0 for i in range(self.env.num_players)] for _ in range(self.num_actor_devices)]

        #position_locks = [threading.Lock() for _ in range(self.env.num_players)]
        position_locks = [threading.Lock() if self.trainable_posi[i] == 1 else 0 for i in range(self.env.num_players)]
        
        for device in range(self.num_actor_devices):
            for i in range(self.num_threads):
                for position in range(self.env.num_players):
                  if self.trainable_posi[position] == 1:
                    thread = threading.Thread(
                        target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,device,position,locks[device][position],position_locks[position]))
                    thread.start()
                    threads.append(thread)

        def checkpoint(frames):
            log.info('Saving checkpoint to %s', self.checkpointpath)
            _agents = learner_model.get_agents()
            torch.save({
                'model_state_dict': [_agents[i].state_dict() if self.trainable_posi[i] == 1 else 0 for i in range(len(_agents))],
                'optimizer_state_dict': [optimizers[i].state_dict() if self.trainable_posi[i] == 1 else 0 for i in range(len(optimizers))],
                "stats": stats,
                'frames': frames,
            }, self.checkpointpath)

            # Save the weights for evaluation purpose
            for position in range(self.env.num_players):
              if self.trainable_posi[position] == 1:
                model_weights_dir = os.path.expandvars(os.path.expanduser(
                    '%s/%s/%s' % (self.savedir, self.xpid, str(position)+'_'+str(frames)+'.pth')))
                torch.save(learner_model.get_agent(position), model_weights_dir)

        timer = timeit.default_timer
        try:
            last_checkpoint_time = timer() - self.save_interval * 60
            while frames < self.total_frames:
                start_frames = frames
                start_time = timer()
                time.sleep(5)

                if timer() - last_checkpoint_time > self.save_interval * 60:
                    checkpoint(frames)
                    last_checkpoint_time = timer()

                end_time = timer()
                fps = (frames - start_frames) / (end_time - start_time)
                log.info('After %i frames: @ %.1f fps Stats:\n%s',
                             frames,
                             fps,
                             pprint.pformat(stats))
        except KeyboardInterrupt:
            return
        else:
            for thread in threads:
                thread.join()
            log.info('Learning finished after %d frames.', frames)

        checkpoint(frames)
        plogger.close()
