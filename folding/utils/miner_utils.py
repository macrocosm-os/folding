
import asyncio
class MinerUtils: 
    def __init__(self):
        self.loop = asyncio.get_event_loop()
    async def check_table(self):
        while True:
            print("checking table")
            await asyncio.sleep(10)

    def start_task(self):
        # schedule the tepeat_task to run and return after 10 seconds 
        asyncio.ensure_future(self.check_table(), loop=self.loop)

    def run_forever(self):
        try:
            # run the event loop indefenitely
            self.loop.run_forever()
        except KeyboardInterrupt:
            print("Shutting down")
        finally:
            # perform cleanup and close the loop
            if self.loop.is_running():
                self.loop.stop()
            self.loop.close()      
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
            print("Shutdown complete")