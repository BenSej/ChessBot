import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.

def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    moveList = []
    moveTree = {}
    if depth == 0:
        return (evaluate(board), moveList, moveTree)

    moves = list(generateMoves(side, board, flags))
    bestMove = moves[0]
    bestList = []

    if side == False:
        heuristic = -math.inf

        for move in moves:
            encodedMove = encode(*move)

            newSide, newBoard, newFlags = makeMove(side, board, move[0], move[1], flags, move[2])
            moveValue, moveList_d1, moveTree_d1 = minimax(newSide, newBoard, newFlags, depth - 1)

            moveTree[encodedMove] = moveTree_d1
            if moveValue > heuristic:
                heuristic = moveValue
                bestMove = move
                bestList = moveList_d1

        moveList = [bestMove] + bestList
        return (heuristic, moveList, moveTree)

    else:
        heuristic = math.inf

        for move in moves:
            encodedMove = encode(*move)

            newSide, newBoard, newFlags = makeMove(side, board, move[0], move[1], flags, move[2])
            moveValue, moveList_d1, moveTree_d1 = minimax(newSide, newBoard, newFlags, depth - 1)

            moveTree[encodedMove] = moveTree_d1
            if moveValue < heuristic:
                heuristic = moveValue
                bestMove = move
                bestList = moveList_d1

        moveList = [bestMove] + bestList
        return (heuristic, moveList, moveTree)


def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    moveList = []
    moveTree = {}
    if depth == 0:
        return (evaluate(board), moveList, moveTree)

    moves = list(generateMoves(side, board, flags))
    bestMove = moves[0]
    bestList = []

    if side == False:
        heuristic = -math.inf

        for move in moves:
            encodedMove = encode(*move)

            newSide, newBoard, newFlags = makeMove(side, board, move[0], move[1], flags, move[2])
            moveValue, moveList_d1, moveTree_d1 = alphabeta(newSide, newBoard, newFlags, depth - 1, alpha, beta)

            moveTree[encodedMove] = moveTree_d1
            if moveValue > heuristic:
                heuristic = moveValue
                bestMove = move
                bestList = moveList_d1

            alpha = max(alpha, heuristic)
            if beta <= alpha:
                break

        moveList = [bestMove] + bestList
        return (heuristic, moveList, moveTree)

    else:
        heuristic = math.inf

        for move in moves:
            encodedMove = encode(*move)

            newSide, newBoard, newFlags = makeMove(side, board, move[0], move[1], flags, move[2])
            moveValue, moveList_d1, moveTree_d1 = alphabeta(newSide, newBoard, newFlags, depth - 1, alpha, beta)


            moveTree[encodedMove] = moveTree_d1

            if moveValue < heuristic:
                heuristic = moveValue
                bestMove = move
                bestList = moveList_d1

            beta = min(beta, heuristic)
            if beta <= alpha:
                break

        moveList = [bestMove] + bestList
        return (heuristic, moveList, moveTree)


def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moveList = []
    moveTree = {}
    moves = list(generateMoves(side, board, flags))
    bestMove = moves[0]
    if side == False:
        heuristic = -math.inf
        for move in moves:
            sum = 0
            encodedMove = encode(*move)
            newSide, newBoard, newFlags = makeMove(side, board, move[0], move[1], flags, move[2])

            for i in range(breadth):
                moveValue, moveTree_d1 = stochasticHelper(newSide, newBoard, newFlags, depth - 1,
                                                                       chooser)
                sum += moveValue

                if encodedMove in moveTree:
                    moveTree[encodedMove].update(moveTree_d1)
                else:
                    moveTree[encodedMove] = moveTree_d1

            average = sum / breadth
            if average > heuristic:
                heuristic = average
                bestMove = move
    else:
        heuristic = math.inf
        for move in moves:
            sum = 0
            encodedMove = encode(*move)
            newSide, newBoard, newFlags = makeMove(side, board, move[0], move[1], flags, move[2])
            for i in range(breadth):
                moveValue, moveTree_d1 = stochasticHelper(newSide, newBoard, newFlags, depth - 1,
                                                                       chooser)
                sum += moveValue
                if encodedMove in moveTree:
                    moveTree[encodedMove].update(moveTree_d1)
                else:
                    moveTree[encodedMove] = moveTree_d1

            average = sum / breadth
            if average < heuristic:
                heuristic = average
                bestMove = move

    return (heuristic, [bestMove], moveTree)

def stochasticHelper(side, board, flags, depth, chooser):
    moveTree = {}
    if depth == 0:
        return (evaluate(board), moveTree)

    moves = list(generateMoves(side, board, flags))
    move = chooser(moves)
    encodedMove = encode(*move)

    newSide, newBoard, newFlags = makeMove(side, board, move[0], move[1], flags, move[2])
    moveValue, moveTree_d1 = stochasticHelper(newSide, newBoard, newFlags, depth - 1, chooser)
    moveTree[encodedMove] = moveTree_d1

    return (moveValue, moveTree)
