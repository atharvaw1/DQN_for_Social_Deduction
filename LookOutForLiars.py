#!/usr/bin/env python

# This sample script connects to the AIWolf server, but
# does not do anything else. It will choose itself as the
# target for any actions requested by the server, (voting,
# attacking ,etc) forcing the server to choose a random target.
import logging
from logging import getLogger, StreamHandler, Formatter, FileHandler
from random import randint
from urllib.parse import parse_qs
import aiwolfpy
import argparse

# name
myname = 'Liar Player'

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

file_handler = FileHandler('aiwolf_game.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(handler_format)
logger.addHandler(file_handler)


class SampleAgent(object):
    
    def __init__(self):
        # my name
        self.base_info = dict()
        self.game_setting = dict()

    def getName(self):
        return self.my_name
    
    # new game (no return)
    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        self.game_setting = game_setting

        # New game init:
        # Store my own ID:
        self.myid = base_info["agentIdx"]
        logger.debug("# INIT: I am agent {}".format(self.myid))
        self.player_total = game_setting["playerNum"]

        # Initialize a list with the hate score for each player
        # Also reduce own-hate score by 10k
        self.player_score = [0]*self.player_total
        self.player_score[self.myid-1] = -10000

        # the liar attribute contains the player ID who has lied the most
        self.liar = self.player_score.index(max(self.player_score)) + 1

        #Statements will be a tuple of (Agent Claim Target) to identify who is making uninformed decisions and who has lied
        self.statements = []

    # new information (no return)
    def update(self, base_info, diff_data, request):
        #base_info = who is dead or alive, my agent id, what day is today
        #diff_data contains one line for each new information received since last diff data, usually talks votes and actions
        self.base_info = base_info
        # At the beginning of the day, reduce score of dead players
        if (request == "DAILY_INITIALIZE"):
            dead = []
            for i in range(self.player_total):
                if (base_info["statusMap"][str(i+1)] == "DEAD"):
                    self.player_score[i] = -10000
                    dead.append(i+1)
            # Check statements to see if anyone labeled as werewolf is dead
            # If accused werewolf is dead but daily initialize still happened accuser is a liar
            for info in self.statements:
                accused = info[2]
                speaker = info[0]
                if (accused in dead):
                    self.player_score[speaker-1] += 1000
                    logger.debug(str(speaker-1) +" lied about " + str(accused) +"being a werewolf!")

        


        # Check each line in Diff Data for talks or votes
        # logging.debug(diff_data)
        for row in diff_data.itertuples():
            type = getattr(row,"type")
            text = getattr(row,"text")
            if (type == "vote"):
                voter = getattr(row,"idx")
                target = getattr(row,"agent")
                # check if this vote has a statement that reasons it 
                # if no statement for reason then add to their lie score a lil bit
                if (not self.validAccuse(target)):
                    logger.debug(str(voter) + " voted for someone with no reason: " + str(target))
                    self.player_score[voter-1] += 100

            elif (type == "talk" and ("VOTE" in text or "WEREWOLF" in text)):
                # telling us someone is werewolf
                source = getattr(row,"agent")
                logger.debug("Telling us someone is werewolf: {}".format(text))
                subjectIndex = text.index("[")
                subjectEndex = text.index("]")
                subjectID = int(text[subjectIndex+1:subjectEndex])
                self.statements.append((int(source), "WEREWOLF", subjectID))
                # Add a little suspicion to someone making a claim
                self.player_score[source-1] += 10


        # Print Current Hate list:
        self.liar = self.player_score.index(max(self.player_score)) + 1
        logger.debug("Hate Score: "+", ".join(str(x) for x in self.player_score))

        
    # Start of the day (no return)
    def dayStart(self):
        return None

    # conversation actions: require a properly formatted
    # protocol string as the return.
    def talk(self):
        hatecycle = [
        "REQUEST ANY (VOTE Agent[{:02d}])",
        "ESTIMATE Agent[{:02d}] WEREWOLF",
        "VOTE Agent[{:02d}]",
        ]
        return hatecycle[randint(0,2)].format(self.liar)
    
    def whisper(self):
        return cf.over()
        
    # targetted actions: Require the id of the target
    # agent as the return
    def vote(self):
        logging.debug("# VOTE: "+str(self.liar))
        return self.liar

    def attack(self):
        logging.debug("# ATTACK: "+str(self.liar))
        return self.liar

    def divine(self):
        return self.liar

    def guard(self):
        logging.debug("# GUARD")
        return self.base_info['agentIdx']

    # Finish (no return)
    def finish(self):
        return None

    def validAccuse(self, target):
        for info in self.statements:
            claim = info[1]
            infoTarget = info[2]
            if (claim == "WEREWOLF" and target == infoTarget):
                return True
        return False    


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
