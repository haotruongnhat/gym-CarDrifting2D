import gym
import numpy as np
import math
import shapely
from shapely.geometry import LineString, Point
from gym.envs.registration import register
import pygame
from pygame import gfxdraw
from lib.dr2_load_data import get_track

WINDOW_W = 1500
WINDOW_H = 1000
FPS = 50  # Frames per second

GLOBAL_SCALING_FACTOR = 0.5
GLOBAL_OFFSET_X = 0 
GLOBAL_OFFSET_Y = 0
INITIAL_POSITION = [0, 0]

LOCAL_SCALING_FACTOR = 5
LOCAL_FOV_RANGE = 50

CAR_LENGTH = 4.37388 / 2
CAR_WIDTH = 1.7 / 2

DEFAULT_MAP = False

if DEFAULT_MAP:

    walls = []

    walls.append([240, 809, 200, 583])
    walls.append([200, 583, 218, 395])
    walls.append([218, 395, 303, 255])
    walls.append([303, 255, 548, 173])
    walls.append([548, 173, 764, 179])
    walls.append([764, 179, 1058, 198])
    walls.append([1055, 199, 1180, 215])
    walls.append([1177, 215, 1220, 272])
    walls.append([1222, 273, 1218, 367])
    walls.append([1218, 367, 1150, 437])
    walls.append([1150, 437, 1044, 460])
    walls.append([1044, 460, 757, 600])
    walls.append([757, 600, 1099, 570])
    walls.append([1100, 570, 1187, 508])
    walls.append([1187, 507, 1288, 443])
    walls.append([1288, 443, 1463, 415])
    walls.append([1463, 415, 1615, 478])
    walls.append([1617, 479, 1727, 679])
    walls.append([1727, 679, 1697, 874])
    walls.append([1694, 872, 1520, 964])
    walls.append([1520, 964, 1100, 970])
    walls.append([1105, 970, 335, 960])
    walls.append([339, 960, 264, 899])
    walls.append([263, 897, 238, 803])
    walls.append([317, 782, 274, 570])
    walls.append([275, 569, 284, 407])
    walls.append([284, 407, 363, 317])
    walls.append([363, 317, 562, 240])
    walls.append([562, 240, 1114, 284])
    walls.append([1114, 284, 1120, 323])
    walls.append([1120, 323, 1045, 377])
    walls.append([1045, 378, 682, 548])
    walls.append([682, 548, 604, 610])
    walls.append([604, 612, 603, 695])
    walls.append([605, 695, 702, 713])
    walls.append([703, 712, 1128, 642])
    walls.append([1129, 642, 1320, 512])
    walls.append([1323, 512, 1464, 497])
    walls.append([1464, 497, 1579, 535])
    walls.append([1579, 535, 1660, 701])
    walls.append([1660, 697, 1634, 818])
    walls.append([1634, 818, 1499, 889])
    walls.append([1499, 889, 395, 883])
    walls.append([395, 883, 330, 838])
    walls.append([330, 838, 315, 782])
    walls.append([319, 798, 306, 725])
    walls.append([276, 580, 277, 543])
    walls.append([603, 639, 622, 590])
    walls.append([599, 655, 621, 704])
    walls.append([1074, 571, 1115, 558])
    walls.append([1314, 516, 1333, 511])
    walls.append([1692, 875, 1706, 830])
    walls.append([277, 912, 255, 872])
    walls.append([1214, 262, 1225, 288])
    walls.append([1601, 470, 1625, 490])
    walls.append([1119, 644, 1139, 634])
    walls.append([687, 710, 719, 710])
    walls.append([1721, 664, 1727, 696])
    walls.append([1015, 392, 1065, 362])
    walls.append([1091, 572, 1104, 568])
    walls.append([1157, 528, 1233, 478])

    map = walls
    reward_gates = [[613, 268, 613, 156], [546, 272, 465, 168], [483, 298, 368, 179], [411, 316, 301, 248], [363, 342, 231, 306], [324, 393, 189, 381], [299, 447, 189, 473], [291, 517, 187, 568], [305, 585, 213, 647], [213, 710, 325, 708], [222, 816, 352, 772], [260, 927, 359, 840], [361, 971, 416, 858], [475, 979, 490, 852], [578, 980, 578, 880], [643, 979, 646, 869], [718, 984, 713, 870], [778, 979, 787, 887], [852, 978, 876, 877], [958, 983, 972, 867], [1040, 976, 1051, 883], [1095, 977, 1126, 860], [1159, 983, 1191, 871], [1222, 980, 1240, 877], [1284, 973, 1297, 877], [1367, 980, 1374, 884], [1452, 975, 1445, 883], [1540, 967, 1507, 873], [1626, 929, 1577, 822], [1716, 835, 1630, 771], [1733, 736, 1646, 703], [1618, 667, 1716, 602], [1598, 611, 1681, 526], [1547, 554, 1597, 441], [1467, 528, 1495, 423], [1392, 529, 1370, 422], [1323, 541, 1256, 450], [1261, 575, 1175, 493], [1155, 642, 1087, 525], [1025, 678, 1026, 557], [923, 699, 930, 569], [807, 707, 841, 600], [701, 711, 746, 627], [611, 657, 720, 591], [719, 509, 809, 571], [862, 542, 813, 480], [932, 521, 919, 445], [1030, 473, 966, 378], [1113, 454, 1065, 364], [1215, 386, 1102, 330], [1099, 298, 1225, 260], [1047, 287, 1087, 191], [949, 288, 958, 187], [856, 284, 854, 179], [761, 275, 759, 167]]
    downScaleFactor = 1.1

    for i in range(len(map)):
        for j in range(len(map[i])):
            map[i][j] = map[i][j]/downScaleFactor
            map[i][0] -= 20
            map[i][2] -= 20

    for i in range(len(reward_gates)):
        for j in range(len(reward_gates[i])):
            reward_gates[i][j] = reward_gates[i][j]/downScaleFactor
            reward_gates[i][0] -= 20
            reward_gates[i][2] -= 20
else:
    path = "map_data\\GR_Argolis_Fourketa Kourva_{}.npz"

    l_pos_x, l_pos_y, l_pos_z, l_progress = get_track(path.format("Left_Drive"))
    r_pos_x, r_pos_y, r_pos_z, r_progress = get_track(path.format("Right_Drive"))

    scaling_factor  = 0.5
    sample_rate = 300

    walls = []

    l_min_x = np.min(l_pos_x)
    l_min_y = np.min(l_pos_y)
    r_min_x = np.min(r_pos_x)
    r_min_y = np.min(r_pos_y)

    GLOBAL_OFFSET_X = np.min([l_min_x, r_min_x])
    GLOBAL_OFFSET_Y = np.min([l_min_y, r_min_y])

    visualized_l_pos_x = (l_pos_x[::sample_rate] - GLOBAL_OFFSET_X)*GLOBAL_SCALING_FACTOR
    visualized_l_pos_y = (l_pos_y[::sample_rate] - GLOBAL_OFFSET_Y)*GLOBAL_SCALING_FACTOR
    visualized_r_pos_x = (r_pos_x[::sample_rate] - GLOBAL_OFFSET_X)*GLOBAL_SCALING_FACTOR
    visualized_r_pos_y = (r_pos_y[::sample_rate] - GLOBAL_OFFSET_Y)*GLOBAL_SCALING_FACTOR
    visualized_l_progress = l_progress[::sample_rate]
    visualized_r_progress = r_progress[::sample_rate]

    visualized_l_pos_x = visualized_l_pos_x.astype(np.int64)
    visualized_l_pos_y = visualized_l_pos_y.astype(np.int64)
    visualized_r_pos_x = visualized_r_pos_x.astype(np.int64)
    visualized_r_pos_y = visualized_r_pos_y.astype(np.int64)

    WINDOW_W = int(np.max([np.max(visualized_l_pos_x), np.max(visualized_r_pos_x)]))
    WINDOW_H = int(np.max([np.max(visualized_l_pos_y), np.max(visualized_r_pos_y)]))

    for idx in range(len(visualized_l_pos_x) -1):
        x1 = visualized_l_pos_x[idx]; y1 = visualized_l_pos_y[idx]
        x2 = visualized_l_pos_x[idx + 1]; y2 = visualized_l_pos_y[idx + 1]
        walls.append([x1, y1, x2, y2])

    for idx in range(len(visualized_r_pos_x) -1):
        x1 = visualized_r_pos_x[idx]; y1 = visualized_r_pos_y[idx]
        x2 = visualized_r_pos_x[idx + 1]; y2 = visualized_r_pos_y[idx + 1]
        walls.append([x1, y1, x2, y2])

    INITIAL_POSITION = [l_pos_x[0], l_pos_y[0]]
    map = walls

class Drifting(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "state_pixels"],
        "render_fps": FPS,
    }

    def __init__(self):
        self.screen = None
        self.map_surf = None
        self.surf = None
        self.clock = None
        self.pos = INITIAL_POSITION #[650, 200]
        self.velX = 0
        self.velY = 0
        self.drag = 0.0
        self.angularVel = 0.0
        self.angularDrag = 0.0
        self.power = 0.7
        self.turnSpeed = 0.04
        self.angle = math.radians(-90)
        self.w = 10
        self.h = 20
        self.on = 0

        self.states = 12
        self.actions = 9

        self.action = [0,0,0]
        self.state = None
        self.reward = None
        self.ray_casting = []

    def _angle_normalize(self, angle):
        # reduce the angle
        pi2 = math.pi*2

        angle =  angle % pi2; 

        # force it to be the positive remainder, so that 0 <= angle < 360  
        angle = (angle + pi2) % pi2;  

        # force into the minimum absolute value residue class, so that -180 < angle <= 180  
        if (angle > math.pi):
            angle -= pi2

        return angle

    def step(self, action):
        self.action = action
        self.turn(action[0])
        self.acc(action[1])

        self.pos[0] += self.velX
        self.pos[1] += self.velY

        self.velX *= self.drag
        self.velY *= self.drag
        self.angle += self.angularVel
        self.angle = self._angle_normalize(self.angle)
        self.angularVel *= self.angularDrag
        reward = -0.01

        ded = False
        # for i in map:
        #     if self.checkCol(i):
        #         ded = True
        #         break
        
        if DEFAULT_MAP:
            if (self.checkCol(reward_gates[self.on])):
                self.on += 1
                reward = 1

            if (self.on > len(reward_gates) - 1):
                self.on = 0

        if (ded):
            reward = -1
            
        state = None #self.getState()
        self.state = state
        self.reward = reward
        
        return state, reward, ded, None

    def _render_global(self):
        for LINE in map:
            gfxdraw.line(self.surf, LINE[0], LINE[1], LINE[2], LINE[3], np.array([0, 0, 0]))

        center_x = (self.pos[0] - GLOBAL_OFFSET_X)*GLOBAL_SCALING_FACTOR
        center_y = (self.pos[1] - GLOBAL_OFFSET_Y)*GLOBAL_SCALING_FACTOR
        gfxdraw.filled_circle(self.surf, int(center_x), int(center_y), 2, np.array([255, 0, 0]))

    def _render_local(self, screen_x, screen_y):
        x, y = self.pos

        ## Get map in fov range
        sample_rate = 10
        l_distance = np.sqrt(np.power(l_pos_x - x, 2) + np.power(l_pos_y - y, 2))
        r_distance = np.sqrt(np.power(r_pos_x - x, 2) + np.power(r_pos_y - y, 2))

        l_in_fov_index = np.where(l_distance < LOCAL_FOV_RANGE)
        r_in_fov_index = np.where(r_distance < LOCAL_FOV_RANGE)

        fov_l_pos_x = l_pos_x[l_in_fov_index]
        fov_l_pos_y = l_pos_y[l_in_fov_index]
        fov_r_pos_x = r_pos_x[r_in_fov_index]
        fov_r_pos_y = r_pos_y[r_in_fov_index]

        ## Offset
        offset_fov_l_pos_x = (fov_l_pos_x - x)*LOCAL_SCALING_FACTOR + LOCAL_FOV_RANGE/2
        offset_fov_l_pos_y = (fov_l_pos_y - y)*LOCAL_SCALING_FACTOR + LOCAL_FOV_RANGE/2
        offset_fov_r_pos_x = (fov_r_pos_x - x)*LOCAL_SCALING_FACTOR + LOCAL_FOV_RANGE/2
        offset_fov_r_pos_y = (fov_r_pos_y - y)*LOCAL_SCALING_FACTOR + LOCAL_FOV_RANGE/2
           
        visualized_fov_l_pos_x = (offset_fov_l_pos_x[::sample_rate]).astype(np.int64) + screen_x
        visualized_fov_l_pos_y = (offset_fov_l_pos_y[::sample_rate]).astype(np.int64) + screen_y
        visualized_fov_r_pos_x = (offset_fov_r_pos_x[::sample_rate]).astype(np.int64) + screen_x
        visualized_fov_r_pos_y = (offset_fov_r_pos_y[::sample_rate]).astype(np.int64) + screen_y
        
        center_point_x = 0 + LOCAL_FOV_RANGE/2
        center_point_x = int(center_point_x) + screen_x
        center_point_y = 0 + LOCAL_FOV_RANGE/2
        center_point_y = int(center_point_y) + screen_y
        gfxdraw.filled_circle(self.surf, center_point_x, center_point_y, 2, np.array([255, 0, 0]))

        for index in range(len(visualized_fov_l_pos_x)-1):
            if (np.abs((visualized_fov_l_pos_x[index] - visualized_fov_l_pos_x[index + 1])) + \
                np.abs((visualized_fov_l_pos_y[index] - visualized_fov_l_pos_y[index + 1]))) > 50:
                continue

            gfxdraw.line(self.surf, visualized_fov_l_pos_x[index], visualized_fov_l_pos_y[index], 
                                    visualized_fov_l_pos_x[index + 1], visualized_fov_l_pos_y[index + 1],
                                    np.array([0, 0, 0]))

            # gfxdraw.filled_circle(self.surf, visualized_fov_l_pos_x[index], visualized_fov_l_pos_y[index], 1, np.array([0, 0, 0]))
        for index in range(len(visualized_fov_r_pos_x)-1):
            if (np.abs((visualized_fov_r_pos_x[index] - visualized_fov_r_pos_x[index + 1])) + \
                np.abs((visualized_fov_r_pos_y[index] - visualized_fov_r_pos_y[index + 1]))) > 50:
                continue
            gfxdraw.line(self.surf, visualized_fov_r_pos_x[index], visualized_fov_r_pos_y[index], 
                                    visualized_fov_r_pos_x[index + 1], visualized_fov_r_pos_y[index + 1],
                                    np.array([0, 0, 0]))
            # gfxdraw.filled_circle(self.surf, visualized_fov_r_pos_x[index], visualized_fov_r_pos_y[index], 1, np.array([0, 0, 0]))

        # Draw car
        
        # LIST = [[CAR_WIDTH, CAR_LENGTH, CAR_WIDTH, -CAR_LENGTH], [-CAR_WIDTH, CAR_LENGTH, -CAR_WIDTH, -CAR_LENGTH], [CAR_WIDTH, CAR_LENGTH, -5, CAR_LENGTH], [5, -CAR_LENGTH, -5, -CAR_LENGTH]]
        LIST = [[CAR_WIDTH, CAR_LENGTH, CAR_WIDTH, -CAR_LENGTH], \
                [-CAR_WIDTH, CAR_LENGTH, -CAR_WIDTH, -CAR_LENGTH], \
                [CAR_WIDTH, CAR_LENGTH, -CAR_WIDTH, CAR_LENGTH], \
                [CAR_WIDTH, -CAR_LENGTH, -CAR_WIDTH, -CAR_LENGTH]]

        HEADING = [0, CAR_LENGTH*0.8]

        verts = []
        for i in LIST:
            LINE = np.array(self.rotatePos(i[0], i[1], i[2], i[3], self.angle)) - np.array([x, y, x, y])
            LINE = (LINE*LOCAL_SCALING_FACTOR + LOCAL_FOV_RANGE/2).astype(np.int64) + np.array([screen_x, screen_y, screen_x, screen_y])
            gfxdraw.line(self.surf, LINE[0], LINE[1], LINE[2], LINE[3], np.array([255, 0, 0]))
            verts.append((LINE[0], LINE[1]))
            verts.append((LINE[2], LINE[3]))

        gfxdraw.aapolygon(self.surf, verts, np.array([255, 0, 0]))
        gfxdraw.filled_polygon(self.surf, verts, np.array([255, 0, 0]))

        LINE = np.array(self.rotatePos(HEADING[0], HEADING[1], HEADING[0], HEADING[1], self.angle)) - np.array([x, y, x, y])
        LINE = (LINE*LOCAL_SCALING_FACTOR + LOCAL_FOV_RANGE/2).astype(np.int64) + np.array([screen_x, screen_y, screen_x, screen_y])
        gfxdraw.filled_circle(self.surf, LINE[0], LINE[1], 2, np.array([0, 255, 0]))



    def render(self, mode: str = "human"):
        import pygame
        from pygame import gfxdraw

        pygame.font.init()

        if self.screen is None:
            pygame.init()
            pygame.display.init()

            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H*2))

        if self.clock is None:
            self.clock = pygame.time.Clock()          
        
        self.surf = pygame.Surface((WINDOW_W, WINDOW_H*2))
        field = [
            (WINDOW_W, WINDOW_H*2),
            (WINDOW_W, -WINDOW_H*2),
            (-WINDOW_W, -WINDOW_H*2),
            (-WINDOW_W, WINDOW_H*2),
        ]
        gfxdraw.filled_polygon(self.surf, field, np.array([102, 204, 102]))

        self._render_global()
        self._render_local(int(WINDOW_W*0.25), int(WINDOW_H*1.25))
      
        font = pygame.font.Font(pygame.font.get_default_font(), 24)
        text = font.render("Pos: {:.2f} - {:.2f}".format(self.pos[0], self.pos[1]), \
                            True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (int(WINDOW_W*0.75), int(WINDOW_H*1.25))
        self.surf.blit(text, text_rect)

        text = font.render("Acceleration: {} - Steering: {}".format(self.action[1], self.action[0]), \
                            True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (int(WINDOW_W*0.75), int(WINDOW_H*1.5))
        self.surf.blit(text, text_rect)

        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        self.screen.fill(0)
        self.screen.blit(self.surf, (0, 0))

        pygame.display.flip()

        return True

    def reset(self):
        self.pos = INITIAL_POSITION
        self.velX = 0
        self.velY = 0
        self.drag = 0.9
        self.angularVel = 0.0
        self.angularDrag = 0.6
        self.power = 0.5
        self.turnSpeed = 0.04
        self.angle = math.radians(-90)
        self.w = 10
        self.h = 20
        self.on = 0

        return True #self.getState()

    def acc(self, value):
        self.velX += math.sin(self.angle) * value * self.power
        self.velY += math.cos(self.angle) * value * self.power

        # if (self.velX > 10):
        #     self.velX = 10

        # if (self.velY > 10):
        #     self.velY = 10

        # if (self.velX < -10):
        #     self.velX = -10

        # if (self.velY < -10):
        #     self.velY = -10

    def turn(self, value):
        self.angularVel += value*self.turnSpeed

    # def right(self, value):
    #     self.angularVel += self.turnSpeed

    # def left(self, value):
    #     self.angularVel -= self.turnSpeed

    def LineInter(self, L1, L2):
        A = (L1[0], L1[1])
        B = (L1[2], L1[3])

        # line 2
        C = (L2[0], L2[1])
        D = (L2[2], L2[3])

        line1 = LineString([A, B])
        line2 = LineString([C, D])

        int_pt = line1.intersection(line2)

        if not int_pt.is_empty:
            point_of_intersection = int_pt.x, int_pt.y
            return point_of_intersection

        return False

    def checkCol(self, line_):
        LIST = [[5, 10, 5, -10], [-5, 10, -5, -10], [5, 10, -5, 10], [5, -10, -5, -10]]
        l = []
        coll = False
        for i in LIST:
            LINE = self.rotatePos(i[0], i[1], i[2], i[3], self.angle)
            inter = self.LineInter(LINE, line_)

            # pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
            #                      ("v2f", (LINE[0], LINE[1], LINE[2], LINE[3]))
            #                      , ('c3B', [50, 255, 30] * 2))
            if not inter == False:
                coll = True

        return coll

    def rotatePos(self, offW, offH, offW2, offH2, angle):
        x = self.pos[0]
        y = self.pos[1]

        X = x + offW
        Y = y + offH

        New_X = x + (X - x) * math.cos(-angle) - (Y - y) * math.sin(-angle)

        New_Y = y + (X - x) * math.sin(-angle) + (Y - y) * math.cos(-angle)

        X = x + offW2
        Y = y + offH2

        New_X2 = x + (X - x) * math.cos(-angle) - (Y - y) * math.sin(-angle)
        New_Y2 = y + (X - x) * math.sin(-angle) + (Y - y) * math.cos(-angle)

        return [New_X, New_Y, New_X2, New_Y2]

    def getState(self):
        range = 10000
        LIST = [[0, 0, range, 0], [0, 0, -range, 0], [0, 0, -range, range], [0, 0, 0, range], [0, 0, range, range]]
        DS = []
        bongs = []
        self.ray_casting = []
        for i in LIST:
            closest = 10000000
            closesIN = []
            LINE = self.rotatePos(i[0], i[1], i[2], i[3], self.angle)
            for LI in map:

                inter = self.LineInter(LINE, LI)

                if not inter == False:
                    distS = math.dist([LINE[0], LINE[1]], [inter[0], inter[1]])
                    if (distS < closest):
                        closest = distS
                        closesIN = inter

            if (len(closesIN) > 0):
                self.ray_casting.append((int(LINE[0]), int(LINE[1]), int(closesIN[0]), int(closesIN[1])))
                bongs.append([LINE[0], LINE[1], closesIN[0], closesIN[1]])
                DS.append(closest)
            else:
                # pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                #                      ("v2f", (LINE[0], LINE[1], LINE[2], LINE[3]))
                #                      , ('c3B', [255, 255, 0] * 2))
                bongs.append([LINE[0], LINE[1], LINE[2], LINE[3]])
                DS.append(-1)

        # ---------------
        if DEFAULT_MAP:
            e = [reward_gates[self.on]]
        else:
            e = []

        for i in bongs:
            closest = 10000000
            closesIN = []
            LINE = i
            for LI in e:

                inter = self.LineInter(LINE, LI)

                if not inter == False:
                    distS = math.dist([LINE[0], LINE[1]], [inter[0], inter[1]])
                    if (distS < closest):
                        closest = distS
                        closesIN = inter

            if (len(closesIN) > 0):
                # pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                #                      ("v2f", (LINE[0], LINE[1], closesIN[0], closesIN[1]))
                #                      , ('c3B', [255, 255, 0] * 2))
                DS.append(closest)
            else:

                DS.append(-1)

        DS.append(self.angularVel)
        vector = np.array([self.velX, self.velY])

        magnitude = np.linalg.norm(vector)
        DS.append(magnitude)
        DS = np.array(DS)
        norm = np.linalg.norm(DS)
        normal_array = DS / norm

        return normal_array

if __name__ == "__main__":
    import pygame
    a = np.array([0.0, 0.0])

    def register_input():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = +1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = -1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[1] = -0.8
                if event.key == pygame.K_RETURN:
                    global restart
                    restart = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[1] = 0

    env = Drifting()
    env.render()

    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, done, info = env.step(a)
            total_reward += r
            # if steps % 200 == 0 or done:
            #     print("\naction " + str([f"{x:+0.2f}" for x in a]))
            #     print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            isopen = env.render()
            if done or restart or isopen is False:
                break
    env.close()

register(
    id='CarDrifting2D-v0',
    entry_point='gym_Drifting2D.Drifting:Drifting'
)