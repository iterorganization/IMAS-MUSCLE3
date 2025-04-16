import logging

from imas_muscle3.data_sink_source import muscled_sink_source

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    muscled_sink_source()
