import asyncio

class MinerUtils: 
    def __init__(self):
        self.loop = asyncio.get_event_loop()

    async def repeat_task():
        while True:
            print("checking table")
            await asyncio.sleep(10)
    def start_task(self):
        # schedule the tepeat_task to run and return after 10 seconds 
        asyncio.ensure_future(self.repeat_task(), loop=self.loop)
    def run_forever(self):
        try:
            # run the event loop indefenitely
            self.loop.run_forever()
        except KeyboardInterrupt:
            print("Shutting down")
        finally:
            # perform cleanup and close the loop
            self.loop.stop()
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
            print("Shutdown complete")
            