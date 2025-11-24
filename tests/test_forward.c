#include "unity.h"
#include "gpt2/layers/attention.h"

void setUp(void) {
    /* This is run before each test */
}

void tearDown(void) {
    /* This is run after each test */
}

void test_forward_basic(void) {
    TEST_ASSERT_EQUAL(1, 1);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_forward_basic);
    return UNITY_END();
}
