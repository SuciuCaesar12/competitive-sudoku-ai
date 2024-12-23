import logging

class Logger:
    def __init__(self, name='sudoku', log_file='team37_A1\\sudokuai.log', level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            #console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            
            # #file handler
            # file_handler = logging.FileHandler(log_file, mode='w')
            # file_handler.setLevel(level)
            # file_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            # self.logger.addHandler(file_handler)
        
    def get_logger(self):
        return self.logger


#helper class to print diagnostics to gives insights in the performance of the algorithm
class Diagnostics:

    def __init__(self, logger, generated_threshold=500, evaluated_threshold=100):
        self.moves_evaluated = 0
        self.moves_generated = 0
        self.diagnostics_moves_evaluated = 0
        self.diagnostics_moves_generated = 0
        
        self.generated_threshold = generated_threshold
        self.evaluated_threshold = evaluated_threshold

        self.logger = logger

    def diagnostics_updated(self):
        # Diagnostics - Log every 500 generated moves that we do
        if (self.moves_generated - self.diagnostics_moves_generated > 10000):
            self.logger.info(f"Generated already {self.moves_generated} moves")
            self.diagnostics_moves_generated = self.moves_generated
        
        # Diagnostics - Log every 100 moves that we do
        if (self.moves_evaluated - self.diagnostics_moves_evaluated > 2000):
            self.logger.info(f"Evaluated already {self.moves_evaluated} moves")
            self.diagnostics_moves_evaluated = self.moves_evaluated

    def moves_generated_increment(self, increment):
        self.moves_generated += increment
        self.diagnostics_updated()
    
    def moves_evaluated_increment(self):
        self.moves_evaluated += 1
        self.diagnostics_updated()

