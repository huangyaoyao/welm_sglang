import unittest

from welm_flash_schedule import MixCohort, decode_schedule, original_gate_inputs, short_decode_final_groups, validate_decode_ranges


def words(ranges, *, fd=False, writers=(), sections=1, mode=0):
    cubes, vectors = 4, 8
    data = [0] * (16 * (1 + sections * (cubes + vectors)))
    data[:7] = [sections, int(fd), 128, 256, cubes, vectors, mode]
    for section, group, start, end in ranges:
        offset = 16 * (1 + section * cubes + group)
        data[offset:offset + 6] = [*start, *end]
    for section, lane, fields in writers:
        offset = 16 * (1 + sections * cubes + section * vectors + lane)
        data[offset:offset + 6] = fields
    return data


def group(core, first, last):
    return 0, core, (first, 0, 0), (last, 0, 0)


def owners(data, *, lengths=(17, 34), need_init=False, query_lengths=(1, 1)):
    return short_decode_final_groups(
        decode_schedule(data), lengths, query_lengths=query_lengths, window=-1, need_init_output=need_init
    )


class FlashScheduleTest(unittest.TestCase):
    def test_changed_schedule_changes_the_owners(self):
        old = words([group(0, 0, 1), group(1, 1, 4)])
        new = words([group(0, 0, 3), group(1, 3, 4)])
        self.assertEqual(owners(old), {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1})
        self.assertEqual(owners(new), {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1})

    def test_requires_whole_groups_and_full_coverage(self):
        for records in ([group(0, 0, 1), group(1, 2, 4)], [group(0, 0, 3), group(1, 2, 4)],
                        [group(0, 0, 3)], [group(0, 0, 5)], [(0, 0, (0, 0, 0), (3, 0, 1))]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                owners(words(records))

    def test_rejects_unsupported_geometry_and_initialization(self):
        data = words([group(0, 0, 4)])
        for options in ({'lengths': (0, 2)}, {'lengths': (257, 2)},
                        {'need_init': None}, {'need_init': True}, {'query_lengths': (1, 0)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                owners(data, **options)
        for descriptor in (words([], mode=1), words([], sections=2), data[:-1], [-1] * 16):
            with self.assertRaises(ValueError):
                owners(descriptor)

    def test_a5_coarse_entry_and_output_only_contract(self):
        # Observed A5 headers, FA half-open ranges and the padded B2 variant.
        def device_descriptor(ranges):
            data = [0] * (16 * (1 + 32 + 64))
            data[:7] = [1, 0, 64, 256, 32, 64, 0]
            for _, group_id, start, end in ranges:
                offset = 16 * (1 + group_id)
                data[offset:offset + 6] = [*start, *end]
            return decode_schedule(data)
        short = device_descriptor([group(i, i, i+1) for i in range(4)])
        padded = device_descriptor([group(0, 0, 1), group(1, 1, 4)])
        contract = MixCohort(32, 64, True, True)
        kwargs = dict(query_lengths=(1, 1), window=-1, need_init_output=True)
        self.assertEqual(short_decode_final_groups(short, [4, 8], cohort=contract, **kwargs),
                         {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3})
        for cohort in (MixCohort(4, 8, True, True), MixCohort(32, 64, False, True),
                       MixCohort(32, 64, True, False), None):
            with self.subTest(cohort=cohort), self.assertRaises(ValueError):
                short_decode_final_groups(short, [4, 8], cohort=cohort, **kwargs)
        with self.assertRaises(ValueError):
            short_decode_final_groups(padded, [65, 0], cohort=contract, **dict(kwargs, query_lengths=(1, 0)))
        with self.assertRaises(ValueError):
            short_decode_final_groups(short, [4, 8], cohort=contract, **dict(kwargs, need_init_output=None))

    def test_original_gate_spans(self):
        for batch in (1, 2, 4, 8):
            groups = {(i, h): i * 2 + h for i in range(batch) for h in range(2)}
            inputs = original_gate_inputs(groups, batch)
            if batch == 8:
                # Flat head 128 splits row 5, KV group 0 between both gates.
                self.assertEqual(inputs, (tuple(range(11)), tuple(range(10, 16))))
            else:
                self.assertEqual(inputs, (tuple(range(2 * batch)),))

    def test_rejects_out_of_range_live_records(self):
        bad_fa = words([(0, 0, (0, 0, 0), (0, 0, 2))])
        bad_fd = words([], fd=True, writers=[(0, 1, (4, 0, 0, 2, 0, 1))])
        bad_m = words([], fd=True, writers=[(0, 1, (0, 0, 0, 2, 11, 2))])
        for descriptor in (bad_fa, bad_fd, bad_m):
            with self.assertRaises(ValueError):
                validate_decode_ranges(decode_schedule(descriptor), (4, 8), (1, 1))

    def test_fd_writer_is_diagnostic_only(self):
        descriptor = words([group(0, 0, 2)], fd=True, writers=[(0, 3, (0, 0, 2, 2, 0, 12))])
        schedule = decode_schedule(descriptor)
        self.assertEqual(schedule.fd_writers, ((0, 3, 0, 0, 2, 2, 0, 12),))
        with self.assertRaisesRegex(ValueError, "FD disabled"):
            owners(descriptor)
        with self.assertRaisesRegex(ValueError, "no live FD"):
            decode_schedule(words([], fd=True))


if __name__ == "__main__":
    unittest.main()
