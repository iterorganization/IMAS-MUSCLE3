.. _`tips and tricks`:

Tips and Tricks
===============

1. You can use the `t_next` `S` port in the accumulator actor to override the standard stopping condition. This might be useful if your workflow contains actors that do not propegate next_timestamp and would otherwise not be compatible with your simulation workflow.
