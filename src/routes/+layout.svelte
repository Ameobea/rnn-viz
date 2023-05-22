<script lang="ts">
  import { maybeInitSentry } from 'src/sentry';

  import 'src/index.css';

  maybeInitSentry();
</script>

<slot />
