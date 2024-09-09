class Card(object):
    colors = {"red","green","purple"}
    numbers = {"1","2","3"}
    shapes = {"squiggle","oval","diamond"}
    fill = {"full","half","empty"}
    def __init__(self, c, n, f, s, image):
        self.color = c
        self.number = n
        self.fill = f
        self.shape = s
        self.image = image
    def get_image(self):
        """
        Returns the image for a card
        
        Parameter:
        Self (Card): This card object
        
        Returns:
        Image: Photo of the card
        """
        return self.image
    def required_traits(self, card):
        """
        For a card input, this returns the required traits for a set.
        If two cards have the same trait, that trait is added to the list of requirements.
        If they have different traits for a type, then the missing one is taken from a list.

        Parameters:
        Self (Card): This card object
        card (Card): A card object to be compared with this card

        Returns: 
        List: A list of required traits for the third card in a set
        """
        traits = set()
        if self.color == card.color:
            traits.add(self.color)
        else:
            traits.update(Card.colors - {self.color, card.color})
        if self.number == card.number:
            traits.add(self.number)
        else:
            traits.update(Card.numbers - {self.number, card.number})
        if self.fill == card.fill:
            traits.add(self.fill)
        else:
            traits.update(Card.fill - {self.fill, card.fill})
        if self.shape == card.shape:
            traits.add(self.shape)
        else:
            traits.update(Card.shapes - {self.shape, card.shape})
        return traits
    def is_set(self, traits):
        """
        checks to see if this card has the same traits as the required ones

        Parameters:
        Self (Card): This card object
        Traits (List): A list of strings describing the required traits
        
        Returns:
        Boolean: whether or not this card has the same traits as the required ones
        """
        self_traits = {self.color, self.number, self.fill, self.shape}
        return self_traits.issuperset(traits)
    def __str__(self):
        return f"color: {self.color}, number: {self.number}, shape: {self.shape}, fill: {self.fill}"
class Game():
    """
    A class used for the logic of playing a game of set
    """
    def find_sets(card_list):
        """
        Loops through all the cards using DFS to find all possible sets. 
        The reason all possible sets are found is that in case a card is mislabeled, some sets
        might be invalid, so showing multiple allows for the user to pick a correct option

        Parameters: 
        card_list (List): A list of card objects to check for possible sets

        Output: 
        List: A list of sets of 3 cards. 
        """
        set_list = []
        alg_cards = card_list.copy()
        for card in card_list:
            alg_cards.remove(card)
            alg_cards_2 = alg_cards.copy()
            for card2 in alg_cards:
                alg_cards_2.remove(card2)
                traits = card.required_traits(card2)
                for card3 in alg_cards_2:
                    if card3.is_set(traits):
                        set_list.append([card, card2, card3])
        return set_list