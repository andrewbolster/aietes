from unittest import TestCase

__author__ = 'bolster'

from aietes import Applications


class TestCommsTrust(TestCase):
    def test_get_metrics_from_empty_packet(self):
        self.assertRaises(KeyError, Applications.CommsTrust.get_metrics_from_received_packet, {})

    def test_get_metrics_from_incomplete_packet(self):
        incomplete_pkt = {
            'tx_pwr_db': 0,
            'rx_pwr_db': 0,
            'delay': 0
        }
        self.assertRaises(KeyError, Applications.CommsTrust.get_metrics_from_received_packet, incomplete_pkt)

    def test_get_metrics_from_packet(self):
        good_pkt = {
            'tx_pwr_db': 0.0,
            'rx_pwr_db': 0.0,
            'delay': 0.0,
            'length': 0.0
        }
        pkt_series_keys = "TXP,RXP,Delay,Length".split(',')
        series = Applications.CommsTrust.get_metrics_from_received_packet(good_pkt)
        self.assertSetEqual(set(series.keys()), set(pkt_series_keys))
