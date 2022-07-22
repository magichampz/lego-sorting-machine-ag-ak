#test
#test 2 lol

import random as rnd
import sys
print(sys.version)

deck = [] # store cards as strings in a list: 5H is Five of Hearts, for example
for suit in ['C', 'H', 'S', 'D']:
    for value in ['2','3','4','5','6','7','8','9','10','J','Q','K','A']:
        deck.append(value+suit)
rnd.shuffle(deck)

hand = []

def drawCard(your_hand):
    card = rnd.randint(0,len(deck))
    your_hand.append(deck[card])
    del deck[card]
    return your_hand

def handValue(hand):
    global handValues
    handValues = 0

    global bust
    bust = False

    for card in hand:
        if card[0].isnumeric():
            handValues += int(card[0])
        elif card[0] in ['J','Q','K']:
            handValues += 10
    
    for card in hand:
        if card[0] == 'A':
            if (handValues+11)<21:
                handValues += 11
            else:
                handValues += 1

    print("\nhand is {}".format(hand))
    print("your current hand value is: " + str(handValues))

    if handValues > 21:

        bust = True
        print("your hand value is {}. you have BUST".format(handValues))

    return handValues

drawCard(hand)
drawCard(hand)

handValue(hand)

while bust == False:
    drawExtraCard = input("Do you want to draw another card? (yes/no) ")
    if drawExtraCard == 'yes' and len(hand) < 5:
        drawCard(hand)
        handValue(hand)
    elif drawExtraCard == 'no' or len(hand) == 5:
        print("\nyour final hand value is: {} ".format(handValues))
        if handValues == 21:
            print("you got BLACKJACK!")
        break
    else:
        print("\ninvalid input\n")
        
        
    





        
            
                         
