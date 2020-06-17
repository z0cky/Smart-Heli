from itertools import cycle
import random
import sys
import pygame
from pygame.locals import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
# Razni parametri, svrha se vidi u imenu
FPS = 60
WIDTH_OF_SCREEN  = 290.0
HEIGHT_OF_SCREEN = 400.0
OBSTACLE_GAP_SIZE  = 100
GROUND_Y = 0.9*HEIGHT_OF_SCREEN
IMAGES = {}
HITBOX = {}
load_saved_pool = 0
save_current_pool = 1
current_pool = []
fitness = []
amount_of_models = 100
next_obstacle_x = -1
next_obstacle_hole_y = -1
generation = 1
max_score=0
current_score=0
max_scores=[]
avg_scores=[]
#ako smo ukljucili da se cuvaju modeli, ova funkcija se aktivira i cuva modele i progres kroz generacije
def save_pool():
    for xi in range(amount_of_models):
        current_pool[xi].save_weights("Modeli/model_new" + str(xi) + ".keras")
    with open("max_and_avg_score.txt","w") as f:
        f.write(str(max_scores)+'\n')
        f.write(str(avg_scores))
    print("Sacuvano")
#ukrstanje, zameni ulazne neurone roditeljima i pritom generisi decu
def crossover(model_i1, model_i2):
    global current_pool
    weights1 = current_pool[model_i1].get_weights()
    weights2 = current_pool[model_i2].get_weights()
    weights_new1 = weights1
    weights_new2 = weights2
    weights_new1[0] = weights2[0]
    weights_new2[0] = weights1[0]
    return np.asarray([weights_new1, weights_new2])
#mutacija, svaki neuron ima 20% sanse da se promeni u krugu -0.5,0.5
def mutate(weights):
    for xi in range(len(weights)):
        for yi in range(len(weights[xi])):
            if random.uniform(0, 1) > 0.8:
                change = random.uniform(-0.5,0.5)
                weights[xi][yi] += change
    return weights
# na osnovu visine helisa, visine prolaza i udaljenosti od prolaza, proceni da li treba skociti ubacujuci u neuronsku mrezu
def predict_action(height, dist, obstacle_height, model_num):
    global current_pool
    height = min(HEIGHT_OF_SCREEN, height) / HEIGHT_OF_SCREEN - 0.5
    dist = dist / 400 - 0.5
    obstacle_height = min(HEIGHT_OF_SCREEN, obstacle_height) / HEIGHT_OF_SCREEN - 0.5
    neural_input = np.asarray([height, dist, obstacle_height])
    neural_input = np.atleast_2d(neural_input)
    output_prob = current_pool[model_num].predict(neural_input, 1)[0]
    if output_prob[0] <= 0.5:
        return 1 # skoci
    return 2  # nemoj
# generisi pocetne modele, 3 ulazna, 3 skrivena, 1 izlazni sloj, koristimo opadajuci gradijent da bismo potpomogli mrezama posto igra nije deterministicka
for i in range(amount_of_models):
    model = Sequential()
    model.add(Dense(output_dim=3, input_dim=3))
    model.add(Activation("sigmoid"))
    model.add(Dense(output_dim=1))
    model.add(Activation("sigmoid"))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=True)
    model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])
    current_pool.append(model)
    fitness.append(-100)
# ako smo izabrali da nastavimo od vec sacuvanog modela
if load_saved_pool:
    for i in range(amount_of_models):
        current_pool[i].load_weights("Modeli/model_new"+str(i)+".keras")
for i in range(amount_of_models):
    print(current_pool[i].get_weights())
# ucitaj slike elemenata igre
HELI_APPEARANCE = (
        'textures/heli1.png',
        'textures/heli2.png',
        'textures/heli3.png',
        'textures/heli4.png'
    )
BACKGROUND = 'textures/buildings.png'
OBSTACLE = 'textures/stone_brick.png'
#postavljamo pozornicu
def main():
    global SCREEN, TICKING_FPS
    pygame.init()
    TICKING_FPS = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((int(WIDTH_OF_SCREEN), int(HEIGHT_OF_SCREEN)))
    pygame.display.set_caption('Smart Heli')
    IMAGES['numbers'] = (
        pygame.image.load('textures/0_score.png').convert_alpha(),
        pygame.image.load('textures/1_score.png').convert_alpha(),
        pygame.image.load('textures/2_score.png').convert_alpha(),
        pygame.image.load('textures/3_score.png').convert_alpha(),
        pygame.image.load('textures/4_score.png').convert_alpha(),
        pygame.image.load('textures/5_score.png').convert_alpha(),
        pygame.image.load('textures/6_score.png').convert_alpha(),
        pygame.image.load('textures/7_score.png').convert_alpha(),
        pygame.image.load('textures/8_score.png').convert_alpha(),
        pygame.image.load('textures/9_score.png').convert_alpha()
    )
    IMAGES['ground'] = pygame.image.load('textures/ground.png').convert_alpha()
    #slike i hitbox-eve elemenata ucitavamo u praktican format zarad lakse obrade, npr okretanja helisa
    while True:
        IMAGES['background'] = pygame.image.load(BACKGROUND).convert()
        IMAGES['player'] = (
            pygame.image.load(HELI_APPEARANCE[0]).convert_alpha(),
            pygame.image.load(HELI_APPEARANCE[1]).convert_alpha(),
            pygame.image.load(HELI_APPEARANCE[2]).convert_alpha(),
            pygame.image.load(HELI_APPEARANCE[3]).convert_alpha(),
        )
        IMAGES['obstacle'] = (
            pygame.transform.rotate(
                pygame.image.load(OBSTACLE).convert_alpha(), 180),
            pygame.image.load(OBSTACLE).convert_alpha(),
        )
        HITBOX['obstacle'] = (
            generate_Hitmask(IMAGES['obstacle'][0]),
            generate_Hitmask(IMAGES['obstacle'][1]),
        )
        HITBOX['player'] = (
            generate_Hitmask(IMAGES['player'][0]),
            generate_Hitmask(IMAGES['player'][1]),
            generate_Hitmask(IMAGES['player'][2]),
            generate_Hitmask(IMAGES['player'][3]),
        )
        # struktura koja belezi pocetno stanje i ciklus helisa
        movementInfo = {
                'playery': int((HEIGHT_OF_SCREEN - IMAGES['player'][0].get_height()) / 2),
                'groundx': 0,
                'playerIndexGen': cycle([0, 1, 2, 3]),
            }
        global fitness
        for idx in range(amount_of_models):
            fitness[idx] = 0
        crashInfo = Game(movementInfo)
        Game_Over(crashInfo)
#ova funkcija obradjuje samu igru
def Game(movementInfo):
    global fitness
    global current_score
    global max_score
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playersXList = []
    playersYList = []
    max_score=0
    current_score=0
    # postavi pocetnu poziciju, brzinu igre i prve prepreke
    for idx in range(amount_of_models):
        playerx, playery = int(WIDTH_OF_SCREEN * 0.2), movementInfo['playery']
        playersXList.append(playerx)
        playersYList.append(playery)
    groundx = movementInfo['groundx']
    groundShift = IMAGES['ground'].get_width() - IMAGES['background'].get_width()
    newObstacle1 = getRandomObstacle()
    newObstacle2 = getRandomObstacle()
    upObstacles = [
        {'x': WIDTH_OF_SCREEN + 200, 'y': newObstacle1[0]['y']},
        {'x': WIDTH_OF_SCREEN + 200 + (WIDTH_OF_SCREEN / 2), 'y': newObstacle2[0]['y']},
    ]
    downObstacles = [
        {'x': WIDTH_OF_SCREEN + 200, 'y': newObstacle1[1]['y']},
        {'x': WIDTH_OF_SCREEN + 200 + (WIDTH_OF_SCREEN / 2), 'y': newObstacle2[1]['y']},
    ]
    global next_obstacle_x
    global next_obstacle_hole_y
    next_obstacle_x = downObstacles[0]['x']
    next_obstacle_hole_y = (downObstacles[0]['y'] + (upObstacles[0]['y'] + IMAGES['obstacle'][0].get_height()))/2
    obstacleVelX = -4
    vertical_velocities = []   
    maximum_speed =  10  
    minimum_speed =  -8  
    gravity = []   
    upwards_boost =  -9  
    flap = [] 
    State = []
    for idx in range(amount_of_models):
        vertical_velocities.append(-9)
        gravity.append(1)
        flap.append(False)
        State.append(True)
    alive_players = amount_of_models
    # pocni igru i vrti dok svi ne umru
    while True:
        # ako je neki helis udrio vrh, oznaci da je mrtav
        for i in range(amount_of_models):
            if playersYList[i] < 0 and State[i] == True:
                alive_players -= 1
                State[i] = False
        # ako su svi mrtvi, prenesi potrebne parametre i izadji iz funkcije, da se obradi game over
        if alive_players == 0:
            return {
                'y': 0,
                'groundCrash': True,
                'groundx': groundx,
                'upObstacles': upObstacles,
                'downObstacles': downObstacles,
                'score': score,
                'playerVelY': 0,
            }
        # sem fitnessa od predjenih prepreka, racunamo i samu duzinu zivota, pogotovo pomaze kod prvih par generacija dok ne uspe da predje prepreku
        for i in range(amount_of_models):
            if State[i] == True:
                fitness[i] += 1
        next_obstacle_x += obstacleVelX
        # za svaki helis proveri da li moze uopste da skace
        for i in range(amount_of_models):
            if State[i] == True:
                if predict_action(playersYList[i], next_obstacle_x, next_obstacle_hole_y, i) == 1:
                    if playersYList[i] > -2 * IMAGES['player'][0].get_height():
                        vertical_velocities[i] = upwards_boost
                        flap[i] = True
        # izadji iz igre ako pritisnemo ESC ili jednostavno izadjemo
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
        # detekcija kolizije za svaki helis
        crashTest = checkCrash({'x': playersXList, 'y': playersYList, 'index': playerIndex},
                               upObstacles, downObstacles)
        # ako je ijedan udrio prepreku, naznaci da je mrtav
        for idx in range(amount_of_models):
            if State[idx] == True and crashTest[idx] == True:
                alive_players -= 1
                State[idx] = False
        # ako su svi mrtvi, resetuj igru
        if alive_players == 0:
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'groundx': groundx,
                'upObstacles': upObstacles,
                'downObstacles': downObstacles,
                'score': score,
                'playerVelY': 0,
            }
        # proveri ako su presli prepreku, dodaj na fitness, dodaj na skor i prebaci se na sledecu
        has_passed=False
        for idx in range(amount_of_models):
            if State[idx] == True:
                obstacle_idx = 0
                playerMidPos = playersXList[idx]
                for obstacle in upObstacles:
                    obstacleMidPos = obstacle['x'] + IMAGES['obstacle'][0].get_width()
                    if obstacleMidPos <= playerMidPos < obstacleMidPos + 4:
                        next_obstacle_x = downObstacles[obstacle_idx+1]['x']
                        next_obstacle_hole_y = (downObstacles[obstacle_idx+1]['y'] + (upObstacles[obstacle_idx+1]['y'] + IMAGES['obstacle'][obstacle_idx+1].get_height())) / 2
                        score += 1
                        current_score+=1
                        has_passed=True
                        fitness[idx] += 50
                    obstacle_idx += 1
        if(has_passed):
            max_score=max_score+1
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        groundx = -((-groundx + 100) % groundShift)
        # promeni brzinu i visinu svakog helisa u zavisnosti od gravitacije i skoka
        for idx in range(amount_of_models):
            if State[idx] == True:
                if vertical_velocities[idx] < maximum_speed and not flap[idx]:
                    vertical_velocities[idx] += gravity[idx]
                if flap[idx]:
                    flap[idx] = False
                playerHeight = IMAGES['player'][playerIndex].get_height()
                playersYList[idx] += min(vertical_velocities[idx], GROUND_Y - playersYList[idx] - playerHeight)
        for uObstacle, lObstacle in zip(upObstacles, downObstacles):
            uObstacle['x'] += obstacleVelX
            lObstacle['x'] += obstacleVelX
        # proveri ako treba da dodamo novu prepreku i resimo se neke koja je daleko iza helisa
        if 0 < upObstacles[0]['x'] < 5:
            newObstacle = getRandomObstacle()
            upObstacles.append(newObstacle[0])
            downObstacles.append(newObstacle[1])
        if upObstacles[0]['x'] < -IMAGES['obstacle'][0].get_width():
            upObstacles.pop(0)
            downObstacles.pop(0)
        SCREEN.blit(IMAGES['background'], (0,0))
        for uObstacle, lObstacle in zip(upObstacles, downObstacles):
            SCREEN.blit(IMAGES['obstacle'][0], (uObstacle['x'], uObstacle['y']))
            SCREEN.blit(IMAGES['obstacle'][1], (lObstacle['x'], lObstacle['y']))
        SCREEN.blit(IMAGES['ground'], (groundx, GROUND_Y))
        # prikazi rezultat i broj zivih helisa
        showScore(max_score,alive_players)
        for idx in range(amount_of_models):
            if State[idx] == True:
                SCREEN.blit(IMAGES['player'][playerIndex], (playersXList[idx], playersYList[idx]))
        pygame.display.update()
        TICKING_FPS.tick(FPS)
# ova funkcija se aktivira kad su svi helisi mrtvi, obradi evoluciju i generisi novu generaciju
def Game_Over(crashInfo):
    global current_pool
    global fitness
    global generation
    global max_scores
    global avg_scores
    new_weights = []
    total_fitness = 0
    # vrsimo klasicnu ruletsku selekciju, gde konvertujemo fitness svakog helisa u percentil sume fitnessa i rolamo da izaberemo roditelje
    for select in range(amount_of_models):
        total_fitness += fitness[select]
    for select in range(amount_of_models):
        fitness[select] /= total_fitness
        if select > 0:
            fitness[select] += fitness[select-1]
    # rolaj da izaberes dva roditelja, mogu biti isti. potom ukrsti i mutiraj generisanu decu, koja se dodaju u sledecu generaciju
    for select in range(int(amount_of_models/2)):
        parent1 = random.uniform(0, 1)
        parent2 = random.uniform(0, 1)
        i1 = -1
        i2 = -1
        for i in range(amount_of_models):
            if fitness[i] >= parent1:
                i1 = i
                break
        for i in range(amount_of_models):
            if fitness[i] >= parent2:
                i2 = i
                break
        new_weights1 = crossover(i1, i2)
        updated_weights1 = mutate(new_weights1[0])
        updated_weights2 = mutate(new_weights1[1])
        new_weights.append(updated_weights1)
        new_weights.append(updated_weights2)
    # nakon sto generises decu, zameni roditelje njima
    for select in range(len(new_weights)):
        fitness[select] = -100
        current_pool[select].set_weights(new_weights[select])
    # ubaci u listu maksimalnog i prosecnog rezultata za tu generaciju
    max_scores=max_scores+[max_score]
    avg_scores=avg_scores+[current_score/(1.00*amount_of_models)]
    print(max_scores)
    print(avg_scores)
    if save_current_pool == 1:
        save_pool()
    generation = generation + 1
    return
# generisi novu prepreku
def getRandomObstacle():
    gapY = random.randrange(0, int(GROUND_Y * 0.6 - OBSTACLE_GAP_SIZE))
    gapY += int(GROUND_Y * 0.1)
    obstacleHeight = IMAGES['obstacle'][0].get_height()
    obstacleX = WIDTH_OF_SCREEN + 10
    return [
        {'x': obstacleX, 'y': gapY - obstacleHeight}, 
        {'x': obstacleX, 'y': gapY + OBSTACLE_GAP_SIZE}, 
    ]
# prikazi rezultat i broj zivih helisa
def showScore(score,players):
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (WIDTH_OF_SCREEN - totalWidth) / 3

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, HEIGHT_OF_SCREEN * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

    scoreDigits = [int(x) for x in list(str(players))]
    totalWidth = 0 

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (WIDTH_OF_SCREEN - totalWidth) / 2 + WIDTH_OF_SCREEN / 3

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, HEIGHT_OF_SCREEN * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()
# pravougaona piksel detekcija kolizije
def checkCrash(players, upObstacles, downObstacles):
    statuses = []
    for idx in range(amount_of_models):
        statuses.append(False)
    for idx in range(amount_of_models):
        statuses[idx] = False
        pi = players['index']
        players['w'] = IMAGES['player'][0].get_width()
        players['h'] = IMAGES['player'][0].get_height()
        if players['y'][idx] + players['h'] >= GROUND_Y - 1:
            statuses[idx] = True
        playerRect = pygame.Rect(players['x'][idx], players['y'][idx],
                      players['w'], players['h'])
        obstacleW = IMAGES['obstacle'][0].get_width()
        obstacleH = IMAGES['obstacle'][0].get_height()
        for uObstacle, lObstacle in zip(upObstacles, downObstacles):
            uObstacleRect = pygame.Rect(uObstacle['x'], uObstacle['y'], obstacleW, obstacleH)
            lObstacleRect = pygame.Rect(lObstacle['x'], lObstacle['y'], obstacleW, obstacleH)
            pHitMask = HITBOX['player'][pi]
            uHitmask = HITBOX['obstacle'][0]
            lHitmask = HITBOX['obstacle'][1]
            uCollide = pixelCollision(playerRect, uObstacleRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lObstacleRect, pHitMask, lHitmask)
            if uCollide or lCollide:
                statuses[idx] = True
    return statuses
def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)
    if rect.width == 0 or rect.height == 0:
        return False
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y
    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False
def generate_Hitmask(image):
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask
if __name__ == '__main__':
    main()
