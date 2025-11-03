class Config:
    # Model parameters
    TEMPORAL_DIM = 512
    HIDDEN_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 6
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-4
    
    # Video processing
    FRAME_SIZE = (224, 224)
    MAX_FRAMES = 100
    FRAME_RATE = 30
    
    # Temporal reasoning
    PREDICTION_HORIZON = 10
    NUM_ACTION_CLASSES = 10
    NUM_EVENT_CLASSES = 20
    NUM_CAUSAL_RELATIONS = 5
    
    # Loss weights
    ACTION_WEIGHT = 1.0
    CAUSAL_WEIGHT = 0.5
    EVENT_WEIGHT = 0.3
    CONSISTENCY_WEIGHT = 0.2
    
    @classmethod
    def to_dict(cls):
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('_') and not callable(value)}