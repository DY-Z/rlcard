import pprint

from rlcard.games.dungeonmayhem import DungeonMayhemClasses


class HumanAgent(object):
    """A human agent for the dungeonmayhem"""

    def __init__(self, num_actions):
        self.use_raw = True
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        """Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by human
        """
        _print_state(state["raw_obs"], state["action_record"], state["legal_actions"])
        action = int(input(">> You choose action (integer): "))
        while action < 0 or action >= len(state["legal_actions"]):
            print("Action illegal...")
            action = int(input(">> Re-choose action (integer): "))
        return state["legal_actions"][action]

    def eval_step(self, state):
        """Predict the action given the curent state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        """
        return self.step(state), {}


def _print_state(raw_obs, action_record, legal_actions):
    """Print out the state of a given player"""
    char = raw_obs
    # print("raw_obs")
    # __import__("pprint").pprint(raw_obs)
    print(
        "Class        :", DungeonMayhemClasses[raw_obs["current_player_idx"]].__name__
    )
    print("Actions Left :", raw_obs["actions"])
    print("Health       :", raw_obs["health"], ", ", raw_obs["immune"])
    for (i, shield) in enumerate(raw_obs["shields"]):
        print(f"Shield {i:02}    :", shield[0], "/", shield[1].shield)
    print("Class        :", DungeonMayhemClasses[raw_obs["current_player_idx"]])
    for i in range(raw_obs["num_players"] - 1):
        targetted = "T" if raw_obs["target"] == raw_obs["others_class"][i] else " "
        print(
            f"- Player {i}{targetted}   : {raw_obs['others_health'][i]:2}",
            ", ",
            raw_obs["others_immune"][i],
            ", ",
            DungeonMayhemClasses[raw_obs["others_class"][i]].__name__,
            sep="",
        )
        for shield in raw_obs["others_shields"][i]:
            print("    Shield   :", shield[0], "/", shield[1].shield)
    for (i, card) in enumerate(raw_obs["hand"]):
        print_card(i, card)

    # print("legal_actions")
    # pprint.pprint([card for card in char["hand"]])


def print_card(i, card, prefix="Card"):
    """Print out a card in a nice form"""
    print(
        f"{prefix:9}{card.id:0>2} {i}: S",
        str(card.shield) + ("I" if card.immune else " "),
        ", D",
        str(card.damage + card.damage_everyone)
        + ("E" if card.damage_everyone else " "),
        ", H",
        card.health,
        ", C",
        card.draw,
        ", A",
        card.actions,
        ", ",
        card.power.__name__ if card.power else "",
        sep="",
    )


def _print_action(action):
    """Print out an action in a nice form

    Args:
        action (str): A string a action
    """
