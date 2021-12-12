from rlcard.games.dungeonmayhem.card import (PaladinDestroyShields,
                                             PaladinGetDiscard)
from rlcard.games.dungeonmayhem.char import DungeonMayhemCharacter


def _new_deck(add):
    """Generate a new deck for the game"""
    add(damage=3)  # ForTheMostJustice(),
    add(damage=3)  # ForTheMostJustice(),
    add(damage=2)  ## ForEvenMoreJustice(),
    add(damage=2)  ## ForEvenMoreJustice(),
    add(damage=2)  ## ForEvenMoreJustice(),
    add(damage=2)  ## ForEvenMoreJustice(),
    add(damage=1, actions=1)  ## ForJustice(),
    add(damage=1, actions=1)  ## ForJustice(),
    add(damage=1, actions=1)  ## ForJustice(),
    add(actions=2)  # FingerwagOfJudgment(),
    add(actions=2)  # FingerwagOfJudgment(),
    add(damage=3, health=1)  # DivineSmite(),
    add(damage=3, health=1)  # DivineSmite(),
    add(damage=3, health=1)  # DivineSmite(),
    add(damage=2, health=1)  # FightingWords(),
    add(damage=2, health=1)  # FightingWords(),
    add(damage=2, health=1)  # FightingWords(),
    add(actions=2)  # HighCharisma(),
    add(actions=2)  # HighCharisma(),
    add(draw=2, health=1)  # CureWounds(),
    add(shield=1, draw=1)  # SpinningParry(),
    add(shield=1, draw=1)  # SpinningParry(),
    add(shield=3)  # DivineShield(),
    add(shield=3)  # DivineShield(),
    add(shield=2)  # Fluffly(),
    add(actions=1, power=PaladinDestroyShields)  # BanishingSmite(),
    add(health=2, power=PaladinGetDiscard)  # DivineInspiration(),
    add(health=2, power=PaladinGetDiscard)  # DivineInspiration(),


class DungeonMayhemPaladin(DungeonMayhemCharacter):
    """docstring for DungeonMayhemPaladin."""

    def __init__(self, np_random):
        super(DungeonMayhemPaladin, self).__init__(np_random)
        self.idx_to_card = DungeonMayhemPaladin.idx_to_card
        self.card_to_idx = DungeonMayhemPaladin.card_to_idx
        self.discardpile = [card for card in DungeonMayhemPaladin.base_deck[0]]

    base_deck = DungeonMayhemCharacter.new_deck(_new_deck)
    card_to_idx = base_deck[1]
    idx_to_card = base_deck[2]
    total_number_of_cards = base_deck[3]
