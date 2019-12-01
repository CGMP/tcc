from app.source.stokedb import StokeDBApp
import logging as log

if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=log.INFO)
    stokeApp = StokeDBApp()
    stokeApp.run()
