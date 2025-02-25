import pygame
import sys

pygame.init()

# -------------------------
# 定义关卡地图
# -------------------------
# 说明：每一行的字符含义如下：
#   '#' 墙
#   ' ' 地板
#   '.' 目标（箱子需要被推到的位置）
#   '$' 箱子
#   '@' 玩家
level_map = [
    "  ####  ",
    "  #  #  ",
    "###  ###",
    "# .@$  #",
    "###  ###",
    "  #  #  ",
    "  ####  "
]

rows = len(level_map)
cols = len(level_map[0])

# -------------------------
# 构建静态层和动态层
# -------------------------
# 静态层：保存墙、地板、目标等信息
# 动态层：保存玩家、箱子等会移动的对象
static_grid = []   # 每个元素为 'wall'、'floor' 或 'goal'
dynamic_grid = []  # 每个元素为 None、'player' 或 'box'
player_pos = None  # 记录玩家所在的 (行, 列) 坐标

for r, line in enumerate(level_map):
    static_row = []
    dynamic_row = []
    for c, ch in enumerate(line):
        if ch == '#':
            static_row.append('wall')
            dynamic_row.append(None)
        elif ch == '.':
            static_row.append('goal')
            dynamic_row.append(None)
        elif ch == '@':
            static_row.append('floor')
            dynamic_row.append('player')
            player_pos = (r, c)
        elif ch == '$':
            static_row.append('floor')
            dynamic_row.append('box')
        else:
            # 空格视为地板
            static_row.append('floor')
            dynamic_row.append(None)
    static_grid.append(static_row)
    dynamic_grid.append(dynamic_row)

# -------------------------
# 初始化 Pygame 窗口
# -------------------------
tile_size = 64
screen_width = cols * tile_size
screen_height = rows * tile_size
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("推箱子游戏")

# 定义颜色
COLOR_WALL   = (100, 100, 100)
COLOR_FLOOR  = (220, 220, 220)
COLOR_GOAL   = (255, 255, 0)
COLOR_BOX    = (160, 82, 45)
COLOR_PLAYER = (0, 0, 255)
COLOR_TEXT   = (255, 0, 0)

font = pygame.font.SysFont(None, 48)

# -------------------------
# 游戏逻辑辅助函数
# -------------------------
def check_win():
    """
    检查是否所有目标位置上都有箱子
    """
    for r in range(rows):
        for c in range(cols):
            if static_grid[r][c] == 'goal' and dynamic_grid[r][c] != 'box':
                return False
    return True

clock = pygame.time.Clock()
game_won = False

# -------------------------
# 游戏主循环
# -------------------------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # 仅在游戏未通关时响应玩家移动
        elif event.type == pygame.KEYDOWN and not game_won:
            # 根据键盘方向键确定移动方向（行, 列的变化）
            dr, dc = 0, 0
            if event.key == pygame.K_UP:
                dr, dc = -1, 0
            elif event.key == pygame.K_DOWN:
                dr, dc = 1, 0
            elif event.key == pygame.K_LEFT:
                dr, dc = 0, -1
            elif event.key == pygame.K_RIGHT:
                dr, dc = 0, 1

            if dr != 0 or dc != 0:
                pr, pc = player_pos
                new_r = pr + dr
                new_c = pc + dc

                # 检查是否越界
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    # 如果目标位置是墙，则不能移动
                    if static_grid[new_r][new_c] == 'wall':
                        pass
                    # 如果目标位置有箱子，则尝试推动箱子
                    elif dynamic_grid[new_r][new_c] == 'box':
                        # 箱子推动后的新位置
                        box_new_r = new_r + dr
                        box_new_c = new_c + dc
                        # 检查箱子推动目标位置是否有效：不出界、不是墙且没有其他箱子
                        if (0 <= box_new_r < rows and 0 <= box_new_c < cols and
                            static_grid[box_new_r][box_new_c] != 'wall' and
                            dynamic_grid[box_new_r][box_new_c] is None):
                            # 推动箱子
                            dynamic_grid[box_new_r][box_new_c] = 'box'
                            # 玩家进入箱子原来的位置
                            dynamic_grid[new_r][new_c] = 'player'
                            dynamic_grid[pr][pc] = None
                            player_pos = (new_r, new_c)
                    # 如果目标位置为空，则直接移动
                    elif dynamic_grid[new_r][new_c] is None:
                        dynamic_grid[new_r][new_c] = 'player'
                        dynamic_grid[pr][pc] = None
                        player_pos = (new_r, new_c)

    # 检查是否通关
    if check_win():
        game_won = True

    # -------------------------
    # 绘制地图
    # -------------------------
    for r in range(rows):
        for c in range(cols):
            rect = pygame.Rect(c * tile_size, r * tile_size, tile_size, tile_size)
            # 绘制静态背景（墙或地板）
            if static_grid[r][c] == 'wall':
                pygame.draw.rect(screen, COLOR_WALL, rect)
            else:
                pygame.draw.rect(screen, COLOR_FLOOR, rect)
                if static_grid[r][c] == 'goal':
                    # 在目标位置上绘制一个小圆，标识目标
                    pygame.draw.circle(screen, COLOR_GOAL, rect.center, tile_size // 4)
            # 绘制动态对象（箱子或玩家）
            if dynamic_grid[r][c] == 'box':
                # 使用略小于格子的矩形绘制箱子
                pygame.draw.rect(screen, COLOR_BOX, rect.inflate(-10, -10))
            elif dynamic_grid[r][c] == 'player':
                pygame.draw.circle(screen, COLOR_PLAYER, rect.center, tile_size // 3)

    # 如果通关，则在屏幕中央显示“你赢了！”提示
    if game_won:
        text = font.render("你赢了!", True, COLOR_TEXT)
        text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
        screen.blit(text, text_rect)

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
sys.exit()
