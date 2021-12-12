from copy import deepcopy
from typing import Any, Optional

import numpy as np

from rlcard.games.dungeonmayhem import (Barbarian, DungeonMayhemClasses,
                                        Paladin, Rogue, Wizard)
from rlcard.games.dungeonmayhem.card import DungeonMayhemCard
from rlcard.games.dungeonmayhem.char import DungeonMayhemCharacter


class DungeonMayhemGame:
    def __init__(self, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.players: list[DungeonMayhemCharacter] = []
        self.losers: list[DungeonMayhemCharacter] = []
        self.history: list[dict] = []
        self.current_player_idx: int = 0
        self.target: Optional[DungeonMayhemCharacter] = None

    def configure(self, game_config):
        """Specifiy some game specific parameters, such as number of players"""

    def init_game(self):
        """Initialize the game"""
        self.wizard = Wizard(self.np_random)
        self.barbarian = Barbarian(self.np_random)
        self.paladin = Paladin(self.np_random)
        self.rogue = Rogue(self.np_random)
        self.players = [char(self.np_random) for char in DungeonMayhemClasses]
        self.losers = []
        for char in self.players:
            char.draw_n(3)
        self.current_player().start_turn()
        self.target = self.calc_target(self.current_player())
        return (self.current_player_state(), self.current_player_idx)

    def play_card(
        self, char: DungeonMayhemCharacter, card: DungeonMayhemCard, decr_actions=True
    ):
        target = self.target
        if decr_actions:
            char.actions -= 1
        char.immune += card.immune
        char.actions += card.actions
        char.heal(card.health)
        char.draw_n(card.draw)
        if card.damage:
            target.take_damage(card.damage)
        if card.damage_everyone:
            for player in self.players:
                player.take_damage(card.damage_everyone)
        if card.power:
            card.power(self, char, target)
        if card.shield:
            char.shields += [(0, card)]
        else:
            char.discardpile.append(card)

        for player in self.players:
            if player.health <= 0 and player not in self.losers:
                self.losers.append(player.__class__.ID)
        if any(
            (player.health <= 0 and player.__class__.ID not in self.losers)
            for player in self.players
        ):
            raise ValueError("All players are dead, but not in losers why")

        if len(char.hand) == 0:
            char.draw_n(2)

    def target_heuristic(
        self, char: DungeonMayhemCharacter, target: DungeonMayhemCharacter
    ):
        """
        Heuristic for choosing a target.
        """
        if char == target:
            return -10000
        health = target.total_health()
        return health  # Maximum health player
        # return -health if health != 0 else -10000  # Minimum health player, but don't target dead players

    def calc_target(self, char: DungeonMayhemCharacter):
        """
        Which player to target by `char`
        """
        return max(self.players, key=lambda x: self.target_heuristic(char, x))

    def current_player(self):
        """Return the current player"""
        return self.players[self.current_player_idx]

    def current_player_state(self):
        """Return the state of the current player"""
        return self.get_state(self.current_player_idx)

    def save_state(self, append_history=True):
        """
        Save the state of the game to an object
        """
        state = {}
        state["wizard"] = deepcopy(self.wizard)
        state["barbarian"] = deepcopy(self.barbarian)
        state["paladin"] = deepcopy(self.paladin)
        state["rogue"] = deepcopy(self.rogue)
        state["losers"] = [loser for loser in self.losers]
        state["current_player"] = self.current_player_idx
        if append_history:
            self.history.append(state)
        return state

    def step_back(self):
        """Return to the previous state of the game

        Returns:
            Status (bool): check if the step back is success or not
        """
        if not self.allow_step_back:
            raise ValueError("Step back is not allowed")
        if len(self.history) > 0:
            state = self.history.pop()
            (self.wizard, self.barbarian, self.paladin, self.rogue) = (
                state["wizard"],
                state["barbarian"],
                state["paladin"],
                state["rogue"],
            )
            self.players = [self.wizard, self.barbarian, self.paladin, self.rogue]
            self.losers = state["losers"]
            return True
        return False

    def step(self, action):
        """
        Take an action in the game

        Args:
        - action: the ID of the card to play

        Returns: (Next players state, Next players id)
        """
        if self.allow_step_back:
            self.save_state(append_history=True)

        char = self.current_player()
        self.target = self.calc_target(char)

        card = char.idx_to_card[action]
        # print("dungeonmayhem/game.py: ", char.hand)
        # print("dungeonmayhem/game.py: ", card)
        char.hand.remove(card)
        self.play_card(char, card)

        if char.actions == 0:  # End of turn
            for _ in range(len(self.players)):
                self.current_player_idx = (
                    self.current_player_idx + 1
                ) % self.get_num_players()
                char = self.current_player()
                if char.health > 0:
                    char.start_turn()
                    break
        elif char.actions < 0:
            raise ValueError("Player has negative actions")

        return (self.current_player_state(), self.current_player_idx)

    NUM_PLAYERS = len(DungeonMayhemClasses)
    NUM_ACTIONS = sum(char.total_number_of_cards for char in DungeonMayhemClasses)

    def get_num_players(self):
        """Return the number of players"""
        return DungeonMayhemGame.NUM_PLAYERS

    def get_num_actions(self):
        """Return the number of actions"""
        return DungeonMayhemGame.NUM_ACTIONS

    def winner(self):
        """Return the winner of the game"""
        # Find the only player in self.players that is not in self.losers
        if self.is_over():
            return next(
                (
                    player
                    for player in self.players
                    if player.__class__.ID not in self.losers
                ),
                None,
            )

    def is_over(self):
        """Check if the game is over"""
        return len(self.losers) >= DungeonMayhemGame.NUM_PLAYERS - 1

    def get_legal_actions(self, player_id):
        """Return the legal actions"""
        char = self.players[player_id]
        #
        # def can_play(card):
        #     # FIXME: may lead to no legal actions
        #     # if char.actions == 2 and card.actions == 2:
        #     #     return False
        #     # TODO: should we prune "useless" cards?
        #     return True
        #
        return [card.id for card in char.hand]
        # return [card.id for card in char.hand if can_play(card)]

    def get_state(self, player_id):
        """Return the stat of the player[player_id]"""
        char = self.players[player_id]
        state: dict[str, Any] = {
            "health": char.health,
            "actions": char.actions,
            "immune": char.immune,
            "hand": char.hand,
            "discardpile": char.discardpile,
            "shields": char.shields,
            "deck": char.deck,
            "target": self.target.__class__.ID,
            # "legal_actions": {card.id: None for card in char.hand},
            # "legal_actions": {i: card.id for (i, card) in enumerate(char.hand)},
            "legal_actions": self.get_legal_actions(player_id),
            "others_health": [
                player.health for player in self.players if player != char
            ],
            "others_immune": [
                player.immune for player in self.players if player != char
            ],
            # TODO: Track others cards more accurately
            "others_discardpile": [
                player.discardpile for player in self.players if player != char
            ],
            "others_class": [
                player.__class__.ID for player in self.players if player != char
            ],
            "others_shields": [
                player.shields for player in self.players if player != char
            ],
            "num_players": self.get_num_players(),
            "current_player_idx": self.current_player_idx,
        }
        return state
