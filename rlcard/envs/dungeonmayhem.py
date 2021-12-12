from itertools import count

import numpy as np

from rlcard.envs import Env
from rlcard.games.dungeonmayhem import DungeonMayhemClasses
from rlcard.games.dungeonmayhem.game import DungeonMayhemGame

counter = count()
state_health_idx = next(counter)
state_shield_idx = next(counter)
state_immune_idx = next(counter)
state_actions_idx = next(counter)
NUM_PLAYERS = DungeonMayhemGame.NUM_PLAYERS
state_others_health_idx = [next(counter) for _ in range(NUM_PLAYERS - 1)]
state_others_shield_idx = [next(counter) for _ in range(NUM_PLAYERS - 1)]
state_others_immune_idx = [next(counter) for _ in range(NUM_PLAYERS - 1)]
state_others_class_idx = [next(counter) for _ in range(NUM_PLAYERS - 1)]
state_total_indices = next(counter)

counter = count()
state_unknown_idx = next(counter)
state_in_hand_idx = next(counter)
state_in_discard_idx = next(counter)
state_in_deck_idx = next(counter)
state_in_shields_idx = [
    next(counter) for _ in range(3)
]  # The sub things are how much remaining shield each card has
state_in_others_discard_idx = [next(counter) for _ in range(NUM_PLAYERS - 1)]
state_in_others_shield_idx = [
    [next(counter) for _ in range(3)] for _ in range(NUM_PLAYERS - 1)
]  # The sub things are how much remaining shield each card has


class DungeonMayhemEnv(Env):
    """
    The environment for Dungeom Mayhem
    """

    def __init__(self, config):
        self.name = "dungeonmayhem"
        self.game = DungeonMayhemGame()
        super(DungeonMayhemEnv, self).__init__(config)
        total_number_of_cards = sum(
            char.total_number_of_cards for char in DungeonMayhemClasses
        )
        self.state_shape = [
            (state_total_indices + total_number_of_cards,)
            for char_class in DungeonMayhemClasses
        ]
        self.action_shape = [None for _ in DungeonMayhemClasses]

    def _extract_state(self, state):
        """Encode state
        Args:
            state (dict): dict of original state
        """
        extracted_state = {}
        """
            extracted_state is a Python dictionary. It consists of 
            observation state['obs'], 
            legal actions state['legal_actions'], 
            raw observation state['raw_obs'],
            raw legal actions state['raw_legal_actions'].
            each of these are np arrays
        """
        obs = np.zeros(self.state_shape[state_health_idx], dtype=int)

        # Information about the current player
        obs[state_health_idx] = state["health"]
        obs[state_shield_idx] = sum(shield[0] for shield in state["shields"])
        obs[state_immune_idx] = state["immune"]
        obs[state_actions_idx] = state["actions"]
        # print('state["hand"]', len(state["hand"]), state["hand"], end="\n\n")
        for card in state["hand"]:
            obs[card.id] = state_in_hand_idx
        # print('state["discardpile"]', state["discardpile"], end="\n\n")
        for card in state["discardpile"]:
            obs[card.id] = state_in_discard_idx
        # print('state["shields"]', state["shields"], end="\n\n")
        for (remaining, card) in state["shields"]:
            obs[card.id] = state_in_shields_idx[remaining]
        # print('state["deck"]', state["deck"], end="\n\n")
        for card in state["deck"]:
            obs[card.id] = state_in_deck_idx

        # Information about other players
        for j in range(DungeonMayhemGame.NUM_PLAYERS - 1):
            i = state["others_class"][j]
            obs[state_others_health_idx[j]] = state["others_health"][j]
            obs[state_others_shield_idx[j]] = sum(
                shield[0] for shield in state["others_shields"][j]
            )
            obs[state_others_immune_idx[j]] = state["others_immune"][j]
            obs[state_others_class_idx[j]] = i
            # TODO: encode other players discard pile and shields
            for card in state["others_discardpile"][j]:
                obs[card.id] = state_in_others_discard_idx[j]
            for (remaining, card) in state["others_shields"][j]:
                obs[card.id] = state_in_others_shield_idx[j][remaining]

        ## TODO: shouldn't this be extracted from state parameter not from self.game? idk
        # extracted_state["legal_actions"] = state["legal_actions"]
        extracted_state["legal_actions"] = {
            action: None for i, action in enumerate(state["legal_actions"])
        }
        extracted_state["raw_legal_actions"] = list(
            i for (i, card) in enumerate(state["legal_actions"])
        )
        # extracted_state["raw_legal_actions"] = [ a for a in extracted_state["legal_actions"] ]
        extracted_state["obs"] = obs
        extracted_state["raw_obs"] = state
        extracted_state["action_record"] = self.action_recorder

        return extracted_state

    def _decode_action(self, action_id):
        """Action id -> the action in the game. Must be implemented in the child class.
        Args:
            action_id (int): the id of the action
        Returns:
            action (string): the action that will be passed to the game engine.
        """
        return action_id

    def get_payoffs(self):
        """Get the payoffs of players. Must be implemented in the child class.
        Returns:
            payoffs (list): a list of payoffs for each player
        """
        # If there is a winner he gets +1, everyone else gets -1
        winner = self.game.winner()
        if winner:
            return np.array(
                [1 if winner == player else -1 for player in self.game.players]
            )
        else:
            return np.zeros(len(self.game.players))

    def _get_legal_actions(self):
        """Get all legal actions for current state
        Returns:
            legal_actions (list): a list of legal actions' id
        """
        # Should this be from the state dictionary? in self.extract_state?
        # Get the IDs of the cards in the current player's hand
        raise NotImplementedError
        return {card.id: None for card in self.game.current_player().hand}

    def get_player_id(self):
        """Get the id of the current player
        Returns:
            player_id (int): the id of the current player
        """
        return self.game.current_player_idx

    # def get_perfect_information(self):
    #     """Get the perfect information of the current state
    #     Returns:
    #         (dict): A dictionary of all the perfect information of the current state
    #     """
