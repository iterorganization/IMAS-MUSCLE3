import sys

ACTORS = f"""
ymmsl_version: v0.2
description: yMMSL configuration for actors exposed by IMAS-MUSCLE3.
programs:
  source_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.source_component
    description: |
      Loads IDSs from a data entry and sends their timeslices out one by
      one on the O_I ports, named `<ids_name>_out`. The IMAS occurrence
      to read is set per port with the `<port_name>_occ` setting, e.g.
      `equilibrium_out_occ`, and defaults to 0.
    supported_settings:
      source_uri: str Mandatory. IMAS URI to load from.
      dd_version: str DD version to convert to. Defaults to the stored
        version.
      iterative: bool Send timeslices one by one, or the whole IDS at
        once. Defaults to true.
      t_min: float Lower bound of the loaded time range.
      t_max: float Upper bound of the loaded time range.
      interpolation_method: str closest, previous or linear. Defaults to
        closest.
  sink_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.sink_component
    description: |
      Saves every IDS received on the F_INIT ports, named
      `<ids_name>_in`, to a data entry. The IMAS occurrence to write is
      set per port with the `<port_name>_occ` setting, e.g.
      `equilibrium_in_occ`, and defaults to 0.
    supported_settings:
      sink_uri: str Mandatory. IMAS URI to save to.
      dd_version: str DD version to convert to. Defaults to the incoming
        version.
      sink_mode: str DBEntry open mode. 'w' overwrites, 'x' refuses to.
        Defaults to 'x'.
      avoid_name_collision: bool On sink_mode 'x', write to a numbered
        path instead of raising when the entry exists. Defaults to true.
  sink_source_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.sink_source_component
    description: |
      Receives IDSs on F_INIT, optionally saves them, and sends out
      stored data for the closest timeslice on O_F. Ports are named
      `<ids_name>_in` and `<ids_name>_out`. The IMAS occurrence to read
      or write is set per port with the `<port_name>_occ` setting, e.g.
      `equilibrium_out_occ`, and defaults to 0.
    supported_settings:
      source_uri: str Mandatory. IMAS URI to load from.
      sink_uri: str IMAS URI to save to. Unset means do not store.
      dd_version: str DD version to convert to. Defaults to the stored
        version.
      sink_mode: str DBEntry open mode. 'w' overwrites, 'x' refuses to.
        Defaults to 'x'.
      avoid_name_collision: bool On sink_mode 'x', write to a numbered
        path instead of raising when the entry exists. Defaults to true.
      interpolation_method: str closest, previous or linear. Defaults to
        closest.
  olc_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.olc_component
    description: |
      Operational Limit Checking through IMAS-Validator on the IDSs
      arriving on the F_INIT ports, named `<ids_name>_in`. Multi-IDS
      validation only for messages sharing a timestamp. Failures are
      reported as HTML and text in the working directory.
    supported_settings:
      rulesets: str Rulesets to run, ';'-separated. Defaults to PDS-OLC.
      extra_rule_dirs: str Extra ruleset directories, ';'-separated.
      apply_generic: bool Also apply the bundled generic rules. Defaults
        to true.
      halt_on_error: bool Exit with an error instead of warning when
        validation fails. Defaults to false.
  accumulator_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.accumulator_component
    description: |
      Accumulates the timeslices arriving on the S ports into one full
      IDS per IDS name and sends those out on O_F, so an actor can run
      serially within a workflow. Ports are named `<ids_name>_in` and
      `<ids_name>_out`, and every input needs a matching output. The
      optional `t_next` S port centralizes the stopping condition.
      Constant or predictable timestepping is advised, since actors
      whose final timeslice cannot be predicted may deadlock.
  iterator_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.iterator_component
    description: |
      Opposite of the accumulator: disassembles the full IDS received on
      each F_INIT port and sends its timeslices out one by one on O_I,
      chaining `next_timestamp`. Ports are named `<ids_name>_in` and
      `<ids_name>_out`, and every input needs a matching output.
    supported_settings:
      time_source_ids: str Input IDS whose time range determines the
        output timeslices. Required when more than one is connected,
        inferred when exactly one is.
      n_timeslices: int Amount of evenly spaced timeslices to send.
        Defaults to the native time array of time_source_ids.
      interpolation_method: str closest, previous or linear. Defaults to
        closest.
  passthrough_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.passthrough_component
    description: |
      Forwards IDSs unchanged, to bridge workflows or bypass an actor
      without editing the coupling. Per IDS name, `<ids_name>_in_f`
      (F_INIT) is forwarded once to `<ids_name>_out_f`, and
      `<ids_name>_in_s` (S) one message at a time to
      `<ids_name>_out_i`. F_INIT takes precedence on O_F when both
      inputs are connected. Startup fails on an output without a
      matching input.
  recorder_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.recorder_component
    description: |
      Sink-only tap on the live traffic of a running workflow: wire it
      as an extra receiver on existing conduits. Each connected S port,
      named `<ids_name>_in`, is an independent timeline, recorded to a
      live-tailable Zarr store at
      `<store_path>/<port>/<iteration_number>.zarr`.
    supported_settings:
      config: str Mandatory. Python file defining the extraction, as an
        extract function or a State class.
      store_path: str Root for the output stores. Defaults to the
        instance's run folder.
      automatic_extract: bool Fall back to BaseState.automatic_extract
        when config's State has no extract. Defaults to false.
      automatic_extract_fields: str Whitespace-separated dotted paths to
        restrict the recording to. Defaults to everything extracted.
  visualization_component:
    executable: {sys.executable}
    args: -u -m imas_muscle3.actors.visualization_component
    description: |
      Live web-based visualization of the IDSs arriving on the S ports,
      using Panel. Plotting logic comes from the `plot_file_path`
      script, which supplies a State and a Plotter class. Ports are
      named `<ids_name>_in` for timeslices and `<ids_name>_md_in` for
      machine description IDSs. Still a prototype.
    supported_settings:
      plot_file_path: str Mandatory. Python script with the State and
        Plotter classes defining the plotting logic.
      port: int Port for the visualization server. Defaults to 0, a
        random available port.
      throttle_interval: float Minimum seconds between plot refreshes.
        Defaults to 0.1.
      keep_alive: bool Leave the server running after the last message,
        instead of stopping it. Defaults to false.
      open_browser: bool Open a browser tab on startup. Defaults to
        true.
      automatic_mode: bool Offer time-dependent quantities through a
        dropdown, picking a plot automatically. Defaults to false.
      automatic_extract_all: bool With automatic_mode, extract all
        time-dependent quantities up front rather than on selection.
        Costly for large IDSs. Defaults to false.
"""
"""yMMSL configuration for all actors exposed by IMAS-MUSCLE3."""
