# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import mozpack.path as mozpath
import buildconfig

def main(output, intgemm_config):
    with open(intgemm_config, "r") as f:
        config = f.read()

    # FIXME detect INTGEMM_COMPILER_SUPPORTS_{AVX2,AVX512BW,AVX512VNNI}
    # Just enable all for now.
    config = config.replace("#cmakedefine", "#define")

    output.write(config)
    output.close()

    return 0
