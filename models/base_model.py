class BaseModel():

    def __init__(self, targets, seq_len, batch_size, epochs):

        self.targets = targets
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs

        self.generator = None
        self.discriminator = None

    def __call__(self, inputs, **kwargs):
        return self.generator(inputs=inputs)

    def build_generator(self):
        return NotImplementedError

    def build_discriminator(self):
        return NotImplementedError
