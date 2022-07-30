<script lang="ts">
  import FunctionComplexityPlot from 'src/components/FunctionComplexityPlot.svelte';
  import functionComplexityDistrs from '../data/functionComplexityDistr.json';
</script>

<div class="root">
  <FunctionComplexityPlot data={functionComplexityDistrs} />
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }
</style>
