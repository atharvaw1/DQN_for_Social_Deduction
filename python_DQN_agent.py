#!/usr/bin/env python

# This sample script connects to the AIWolf server, but
# does not do anything else. It will choose itself as the
# target for any actions requested by the server, (voting,
# attacking ,etc) forcing the server to choose a random target.
import logging
import json
from logging import getLogger, StreamHandler, Formatter, FileHandler
import aiwolfpy
import argparse
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
from DQN_network import DQNetwork
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




# name
myname = 'DQN player'#+ str(random.randint(1,100))

# content factory
cf = aiwolfpy.ContentFactory()

# logger
logger = getLogger("aiwolfpy")
logger.setLevel(logging.NOTSET)
# handler
stream_handler = StreamHandler()
stream_handler.setLevel(logging.NOTSET)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(handler_format)


logger.addHandler(stream_handler)




class SampleAgent(object):


    def __init__(self):
        # my name
        self.base_info = dict()
        self.game_setting = dict()
        self.diff_data = dict()




    def getName(self):
        return self.my_name
    
    # new game (no return)
    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        self.game_setting = game_setting

        self.myid = base_info["agentIdx"]
        self.myrole = base_info["myRole"]

        self.player_total = game_setting["playerNum"]

        self.all_states = []
        self.all_actions = []
        self.all_rewards = []


        #### Arrays that stores attitude, declaration and fact of the players

        self.declaration = np.array(self.player_total*["unknown"],dtype="object")
        if self.myrole == "VILLAGER" or self.myrole == "SEER":
            self.declaration[self.myid-1] = "villager"
        if self.myrole == "WEREWOLF" or self.myrole == "POSSESSED":
            self.declaration[self.myid-1] = "werewolf"

        #Player n fact (confirmed role of player n) , value is one-hot encoded for each role [unknown,werewolf,villager,seer, dead]
        self.fact = np.array(["unknown"],dtype="object")
        if self.myrole == "VILLAGER"  or self.myrole == "SEER":
            self.fact[0] = "villager"
        if self.myrole == "WEREWOLF" or self.myrole == "POSSESSED":
            self.fact[0] = "werewolf"

        # Attitude of player n towards player m , value is one-hot encoded for [neutral,positive,negative]
        self.attitude = np.array([self.player_total*["neutral"]]*self.player_total,dtype="object")

        # print(self.declaration)
        # print(self.fact)
        # print(self.attitude.flatten())

        self.dlabels = OneHotEncoder().fit(np.array(["unknown","werewolf","villager","dead"]).reshape(-1,1))
        self.flabels = OneHotEncoder().fit(np.array(["werewolf","villager"]).reshape(-1,1))
        self.alabels = OneHotEncoder().fit(np.array(["neutral","positive","negative"]).reshape(-1,1))


        attitude_enc = self.alabels.transform(self.attitude.flatten().reshape(-1,1))
        fact_enc = self.flabels.transform(self.fact.reshape(-1,1))
        declaration_enc = self.dlabels.transform(self.declaration.reshape(-1,1))
        temp = np.append(declaration_enc.toarray(),fact_enc.toarray())
        self.game_state = np.append(temp,attitude_enc.toarray())

        self.sess = tf.Session()

        self.DQNetwork = DQNetwork(self.game_state.shape,
                                    self.player_total,
                                    0.001)
#         self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if self.myrole in ["WEREWOLF","POSSESSED"]:
            saver.restore(self.sess, "./models/model_evil.ckpt")
        else:
            saver.restore(self.sess, "./models/model.ckpt")




    def predict_action(self,state):
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        explore_probability = np.random.rand()
        #Explore 10% of the time
        exp_exp_tradeoff = 0.0

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            action = random.randint(0,self.player_total-1)

        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = self.sess.run(self.DQNetwork.output, feed_dict = {self.DQNetwork.inputs_: state.reshape((1, *state.shape))})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = int(choice)

        return action
        
    # new information (no return)
    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        self.diff_data = diff_data
#         print(diff_data)
#         Update dead players
        if (request == "DAILY_INITIALIZE"):
            for i in range(self.player_total):
                if (base_info["statusMap"][str(i+1)] == "DEAD"):
                    self.declaration[i] = "dead"


        for row in diff_data.itertuples():
            type = getattr(row,"type")
            text = getattr(row,"text")

            if (type == "vote"):
                #Update attitude values for player voted against
                voter = getattr(row,"idx")
                target = getattr(row,"agent")
                self.attitude[voter-1,target-1] = "negative"

            elif (type == "talk"):
                source = getattr(row,"agent")



                if "COMINGOUT" in text:
                    if "SEER" in text or "VILLAGER" in text:
                        self.declaration[source-1] = "villager"
                    if "WEREWOLF" in text:
                        self.declaration[source-1] = "werewolf"

                if "ESTIMATE" in text:
                    if "SEER" in text or "VILLAGER" in text:
                        m = re.findall('\[..\\]',text)
                        target = int(m[-1][1:-1])
                        self.attitude[source-1,target-1] = "positive"
                    if "WEREWOLF" in text or "POSSESSED" in text:
                        m = re.findall('\[..\\]',text)
                        target = int(m[-1][1:-1])
                        self.attitude[source-1,target-1] = "negative"


                if "DIVINED" in text and "HUMAN" in text:
                    m = re.findall('\[..\\]',text)
                    target = int(m[-1][1:-1])
                    self.attitude[source-1,target-1] = "positive"
                    self.declaration[source-1] = "villager"
                if "DIVINED" in text and "WEREWOLF" in text:
                    m = re.findall('\[..\\]',text)
                    target =int(m[-1][1:-1])
                    self.attitude[source-1,target-1] = "negative"
                    self.declaration[source-1] = "villager"



        attitude_enc = self.alabels.transform(self.attitude.flatten().reshape(-1,1))
        fact_enc = self.flabels.transform(self.fact.reshape(-1,1))
        declaration_enc = self.dlabels.transform(self.declaration.reshape(-1,1))
        temp = np.append(declaration_enc.toarray(),fact_enc.toarray())
        self.game_state = np.append(temp,attitude_enc.toarray())


        
    # Start of the day (no return)
    def dayStart(self):
        return None

    # conversation actions: require a properly formatted
    # protocol string as the return.
    def talk(self):
        # Try to get other to vote your pick
        hatecycle = [
                "REQUEST ANY (VOTE Agent[{:02d}])",
                "ESTIMATE Agent[{:02d}] WEREWOLF",
                "VOTE Agent[{:02d}]",
                ]
        if self.myrole in ["WEREWOLF","POSSESSED"]:
            target = self.predict_action(self.game_state)+1
            while target+1!= self.myid:
                target = random.randint(0,self.player_total-1)
            return hatecycle[random.randint(0,2)].format(target)
        else:
            return "OVER"
    
    def whisper(self):
        #Same as talk
        hatecycle = [
            "REQUEST ANY (ATTACK Agent[{:02d}])",
            "ATTACK Agent[{:02d}]",
            ]
        return hatecycle[random.randint(0,2)].format(self.predict_action(self.game_state)+1)
        
    # targeted actions: Require the id of the target
    # agent as the return
    def vote(self):
        target = self.predict_action(self.game_state)
        self.all_states.append(self.game_state)
        action = [0]*self.player_total
        action[target] = 1
        self.all_actions.append(action)

        if self.declaration[target] == "dead" or target+1 == self.myid:
            self.all_rewards.append(-1)
        else:
            self.all_rewards.append(-0.1)

        return target+1

    def attack(self):
        target = self.predict_action(self.game_state)
        self.all_states.append(self.game_state)
        action = [0]*self.player_total
        action[target] = 1
        self.all_actions.append(action)

        if self.declaration[target] == "dead" or target+1 == self.myid:
            self.all_rewards.append(-1)
        else:
            self.all_rewards.append(-0.1)
        return target+1

    def divine(self):
        return self.base_info['agentIdx']

    def guard(self):
        return self.base_info['agentIdx']


    def update_weights(self,reward):
        saver = tf.train.Saver()
        Qs_next_state = self.sess.run(self.DQNetwork.output, feed_dict = {self.DQNetwork.inputs_: np.array(self.all_states)})

        target_Qs_batch = []
        for i in range(0, len(self.all_states)):
            # If we are in a terminal state, only equals reward
            if i==len(self.all_states)-1:
                target_Qs_batch.append(reward)
            else:
                target =  self.all_rewards[i] + 0.99 * np.max(Qs_next_state[i+1])
                target_Qs_batch.append(target)

        targets_mb = np.array([each for each in target_Qs_batch])
        print(targets_mb)

        loss, _ = self.sess.run([self.DQNetwork.loss, self.DQNetwork.optimizer],
                                            feed_dict={self.DQNetwork.inputs_: np.array(self.all_states),
                                                       self.DQNetwork.target_Q: targets_mb,
                                                       self.DQNetwork.actions_: np.array(self.all_actions)})
        print(loss)

        if self.myrole in ["WEREWOLF","POSSESSED"]:
            save_path = saver.save(self.sess, "./models/model_evil.ckpt")
        else:
            save_path = saver.save(self.sess, "./models/model.ckpt")

        self.sess.close()


    # Finish (no return)
    def finish(self):

        #If any werewolf is alive they won
        for row in self.diff_data.itertuples():
            agent = getattr(row,"agent")
            text = getattr(row,"text")
            if "WEREWOLF" in text and self.base_info["statusMap"][str(agent)] == "ALIVE":
                #print("WEREWOLF won!!")
                if self.myrole in ["VILLAGER","SEER"]:
                    reward = -10
                else:
                    reward = 10
                self.all_rewards.append(reward)
#                 self.update_weights(reward)
                return None



        #If no werewolf is alive humans won
        #print("HUMAN won!!")
        if self.myrole in ["VILLAGER","SEER"]:
            reward = 10
        else:
            reward = -10
        self.all_rewards.append(reward)
#         self.update_weights(reward)

        return None
    

agent = SampleAgent()

# read args
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-p', type=int, action='store', dest='port')
parser.add_argument('-h', type=str, action='store', dest='hostname')
parser.add_argument('-r', type=str, action='store', dest='role', default='none')
parser.add_argument('-n', type=str, action='store', dest='name', default=myname)
input_args = parser.parse_args()


client_agent = aiwolfpy.AgentProxy(
    agent, input_args.name, input_args.hostname, input_args.port, input_args.role, logger, "pandas"
)

# run
if __name__ == '__main__':
    client_agent.connect_server()
