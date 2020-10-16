import numpy as np

class coin:

    def __init__(self):
        self.head = 0
        self.tail = 0

    def flip(self):
        if (np.random.randint(0,2,dtype=int) == 1): self.head += 1 
        # randint lower bound is inclusive, higher bound is exclusive
        else: self.tail += 1


def simulation():
    coin_list = np.zeros(3)
    randint_ = np.random.randint(1,1000)
    lowest_head_freq = 10
    for num in range(1000):
        coin_ = coin()
        for _ in range(10):
            coin_.flip()

    # if (num == 1):
    #     coin_list[0] = coin_.head
    # elif (num == randint_):
    #     coin_list[1] = coin_.head
        if (coin_.head < lowest_head_freq):
            lowest_head_freq = coin_.head

    coin_list[2] = lowest_head_freq
    return coin_list
        
def main():
    dataset = []
    for _ in range(100):
        coin_list = simulation()
        dataset.extend(coin_list)
        # print(_)
    print(sum(dataset)/1000)

if __name__ == "__main__":
    main()