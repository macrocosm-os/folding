import bittensor as bt
btlogger = bt.btlogging.LoggingMachine(config=bt.config(), name="Miner logger")
btlogger.on()