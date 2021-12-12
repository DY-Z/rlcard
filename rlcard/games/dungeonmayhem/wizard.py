from rlcard.games.dungeonmayhem.card import WizardStealShield, WizardSwapHP
from rlcard.games.dungeonmayhem.char import DungeonMayhemCharacter


def _new_deck(add):
    """Generate a new deck for the game"""
    add(damage=3)  # LightningBolt(),
    add(damage=3)  # LightningBolt(),
    add(damage=3)  # LightningBolt(),
    add(damage=3)  # LightningBolt(),
    add(damage=2)  # BurningHands(),
    add(damage=2)  # BurningHands(),
    add(damage=2)  # BurningHands(),
    add(damage=1, actions=1)  # MagicMissile(),
    add(damage=1, actions=1)  # MagicMissile(),
    add(damage=1, actions=1)  # MagicMissile(),
    add(actions=2)  # SpeedOfThought(),
    add(actions=2)  # SpeedOfThought(),
    add(actions=2)  # SpeedOfThought(),
    add(health=1, actions=1)  # EvilSneer(),
    add(health=1, actions=1)  # EvilSneer(),
    add(draw=3)  # KnowledgeIsPower(),
    add(draw=3)  # KnowledgeIsPower(),
    add(draw=3)  # KnowledgeIsPower(),
    add(shield=1, draw=1)  # Shield(),
    add(shield=1, draw=1)  # Shield(),
    add(shield=2)  # Stoneskin(),
    add(shield=3)  # MirrorImage(),
    add(damage_everyone=3)  # Fireball(),
    add(damage_everyone=3)  # Fireball(),
    add(power=WizardStealShield)  # Charm(),
    add(power=WizardStealShield)  # Charm(),
    add(power=WizardSwapHP)  # VampiricTouch(),
    add(power=WizardSwapHP)  # VampiricTouch(),


class DungeonMayhemWizard(DungeonMayhemCharacter):
    """docstring for DungeonMayhemWizard."""

    def __init__(self, np_random):
        super(DungeonMayhemWizard, self).__init__(np_random)
        self.idx_to_card = DungeonMayhemWizard.idx_to_card
        self.card_to_idx = DungeonMayhemWizard.card_to_idx
        self.discardpile = [card for card in DungeonMayhemWizard.base_deck[0]]

    base_deck = DungeonMayhemCharacter.new_deck(_new_deck)
    card_to_idx = base_deck[1]
    idx_to_card = base_deck[2]
    total_number_of_cards = base_deck[3]
