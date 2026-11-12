from synneon.input.compass import Compass


class TestCompassZoneCalculation:
    def test_initial_group_is_full_alphabet(self):
        c = Compass()
        assert c.current_group == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def test_initial_zone_sizes_26_letters(self):
        """ceil(26/4)=7 → UP=7, RIGHT=7, DOWN=7, LEFT=5."""
        c = Compass()
        assert len(c.mapped_zones["up"]) == 7
        assert len(c.mapped_zones["right"]) == 7
        assert len(c.mapped_zones["down"]) == 7
        assert len(c.mapped_zones["left"]) == 5

    def test_initial_up_zone_content(self):
        c = Compass()
        assert c.mapped_zones["up"] == list("ABCDEFG")

    def test_initial_right_zone_content(self):
        c = Compass()
        assert c.mapped_zones["right"] == list("HIJKLMN")

    def test_initial_down_zone_content(self):
        c = Compass()
        assert c.mapped_zones["down"] == list("OPQRSTU")

    def test_initial_left_zone_content(self):
        c = Compass()
        assert c.mapped_zones["left"] == list("VWXYZ")

    def test_zones_property_aliases_mapped_zones(self):
        c = Compass()
        assert c.zones is c.mapped_zones

    def test_current_text_property_aliases_text_buffer(self):
        c = Compass()
        assert c.current_text == c.text_buffer == ""


class TestCompassProcessInput:
    def test_drill_into_up_narrows_group(self):
        c = Compass()
        c.process_input("up")
        assert c.current_group == list("ABCDEFG")

    def test_drill_second_level_up(self):
        """After drilling into A-G (7), quarter=2 → up zone = [A, B]."""
        c = Compass()
        c.process_input("up")  # now in A-G
        c.process_input("up")  # now in A-B
        assert c.current_group == list("AB")
        assert c.mapped_zones["up"] == ["A"]
        assert c.mapped_zones["right"] == ["B"]

    def test_single_letter_zone_types_and_resets(self):
        c = Compass()
        c.process_input("up")  # A-G
        c.process_input("up")  # A-B
        c.process_input("up")  # A is single → types A, resets
        assert c.text_buffer == "A"
        assert c.current_group == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def test_typing_appends_to_buffer(self):
        c = Compass()
        # Type two letters
        c.process_input("up")  # A-G
        c.process_input("up")  # A-B
        c.process_input("up")  # types A
        c.process_input("up")  # A-G again
        c.process_input("up")  # A-B
        c.process_input("right")  # types B
        assert c.text_buffer == "AB"

    def test_history_stack_grows_on_drill(self):
        c = Compass()
        c.process_input("up")
        assert len(c.history_stack) == 1
        c.process_input("up")
        assert len(c.history_stack) == 2

    def test_reset_after_typing_clears_history(self):
        c = Compass()
        c.process_input("up")  # drill
        c.process_input("up")  # drill
        c.process_input("up")  # types A, resets
        assert len(c.history_stack) == 0

    def test_invalid_direction_ignored(self):
        c = Compass()
        c.process_input("diagonal")
        assert c.current_group == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert c.text_buffer == ""

    def test_empty_zone_ignored(self):
        """After drilling to A-B, DOWN and LEFT zones are empty."""
        c = Compass()
        c.process_input("up")  # A-G
        c.process_input("up")  # A-B
        before_group = list(c.current_group)
        c.process_input("down")  # empty zone — no change
        assert c.current_group == before_group

    def test_right_drill_navigates_correctly(self):
        c = Compass()
        c.process_input("right")  # H-N
        assert c.current_group == list("HIJKLMN")

    def test_down_drill_navigates_correctly(self):
        c = Compass()
        c.process_input("down")  # O-U
        assert c.current_group == list("OPQRSTU")

    def test_left_drill_navigates_correctly(self):
        c = Compass()
        c.process_input("left")  # V-Z
        assert c.current_group == list("VWXYZ")


class TestCompassBack:
    def test_back_at_top_level_no_text_is_noop(self):
        c = Compass()
        c.back()
        assert c.text_buffer == ""
        assert c.current_group == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def test_back_navigates_up_from_drill(self):
        c = Compass()
        c.process_input("up")  # drill into A-G
        c.back()
        assert c.current_group == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def test_back_pops_history_correctly(self):
        c = Compass()
        c.process_input("up")  # level 1: A-G
        c.process_input("up")  # level 2: A-B
        c.back()
        assert c.current_group == list("ABCDEFG")
        c.back()
        assert c.current_group == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def test_back_at_top_with_text_deletes_last_char(self):
        c = Compass()
        c.process_input("up")  # A-G
        c.process_input("up")  # A-B
        c.process_input("up")  # types A
        assert c.text_buffer == "A"
        c.back()
        assert c.text_buffer == ""

    def test_back_deletes_one_char_at_a_time(self):
        c = Compass()
        # Type A twice
        for _ in range(2):
            c.process_input("up")
            c.process_input("up")
            c.process_input("up")
        assert c.text_buffer == "AA"
        c.back()
        assert c.text_buffer == "A"


class TestCompassReset:
    def test_reset_compass_clears_navigation(self):
        c = Compass()
        c.process_input("up")
        c.reset_compass()
        assert c.current_group == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert c.history_stack == []

    def test_reset_compass_preserves_text(self):
        c = Compass()
        c.process_input("up")
        c.process_input("up")
        c.process_input("up")  # types A
        c.process_input("up")
        c.reset_compass()
        assert c.text_buffer == "A"

    def test_clear_resets_both_text_and_navigation(self):
        c = Compass()
        c.process_input("up")
        c.process_input("up")
        c.process_input("up")  # types A
        c.process_input("up")  # drill into A-G again
        c.clear()
        assert c.text_buffer == ""
        assert c.current_group == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert c.history_stack == []
